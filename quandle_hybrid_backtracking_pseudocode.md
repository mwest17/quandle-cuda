# Hybrid quandle backtracking: CPU + CUDA pseudocode

This file captures a full backtracking design for searching valid finite quandles using a hybrid CPU/GPU strategy.

---

## 1. State representation

```cpp
struct Cell {
    int row;
    int col;
};

struct State {
    int n;                  // quandle order
    int depth;              // number of assigned entries
    int assignedCount;      // same as depth, for convenience
    uint8_t* table;         // flat array of size n*n, values in [0, n-1]
    uint8_t* assigned;      // flat array of size n*n, 1 if assigned
    bool valid;
    bool complete;
};

struct Result {
    enum Kind { INVALID, PARTIAL_VALID, COMPLETE_VALID };
    Kind kind;
    State state;
};
```

Recommended implementation detail:

- Use a flat array for the multiplication table to keep CUDA kernels simple.
- Pre-allocate memory for worst-case output buffers, because dynamic allocation inside the kernel is expensive and risky.
- Store only a compact bitmask or packed byte array for each state if memory pressure becomes severe.

---

## 2. Top-level search loop on CPU

```cpp
void searchHybrid()
{
    constexpr int GPU_BATCH_SIZE = 1024;
    constexpr int GPU_TRIGGER = 1024 * 10;

    std::vector<State> stack;
    std::unordered_set<std::string> seenComplete;

    std::vector<State> roots = buildInitialRoots();

    for (State& root : roots)
    {
        if (isFeasible(root))
        {
            stack.push_back(root);
        }
    }

    while (!stack.empty())
    {
        if ((int)stack.size() > GPU_TRIGGER)
        {
            std::vector<State> batch;
            for (int i = 0; i < GPU_BATCH_SIZE && !stack.empty(); ++i)
            {
                batch.push_back(stack.back());
                stack.pop_back();
            }

            std::vector<Result> gpuResults = runGpuBatch(batch);

            for (const Result& r : gpuResults)
            {
                if (r.kind == Result::INVALID)
                    continue;

                if (r.kind == Result::COMPLETE_VALID)
                {
                    std::string key = canonicalize(r.state);
                    if (!seenComplete.count(key))
                    {
                        seenComplete.insert(key);
                        recordCompleteState(r.state);
                    }
                    continue;
                }

                if (r.kind == Result::PARTIAL_VALID)
                {
                    stack.push_back(r.state);
                }
            }
        }
        else
        {
            State s = stack.back();
            stack.pop_back();

            if (!isFeasible(s))
                continue;

            if (isComplete(s))
            {
                if (isValidQuandle(s))
                {
                    std::string key = canonicalize(s);
                    if (!seenComplete.count(key))
                    {
                        seenComplete.insert(key);
                        recordCompleteState(s);
                    }
                }
                continue;
            }

            Cell cell = chooseNextUnassignedCell(s);
            std::vector<int> candidates = generateCandidates(s, cell);

            for (int value : candidates)
            {
                State child = applyAssignment(s, cell, value);
                if (isFeasible(child))
                {
                    stack.push_back(child);
                }
            }
        }
    }
}
```

---

## 3. Feasibility check on CPU

```cpp
bool isFeasible(const State& s)
{
    // 1. Check every assigned entry against the quandle axioms.
    for (int i = 0; i < s.n; ++i)
    {
        for (int j = 0; j < s.n; ++j)
        {
            if (!isAssigned(s, i, j))
                continue;

            if (!axiom1Holds(s, i, j))
                return false;

            if (!axiom2Holds(s, i, j))
                return false;

            if (!axiom3HoldsPartial(s, i, j))
                return false;
        }
    }

    // 2. Check uniqueness in every row and every column.
    for (int r = 0; r < s.n; ++r)
    {
        std::vector<int> seen(s.n, 0);
        for (int c = 0; c < s.n; ++c)
        {
            if (!isAssigned(s, r, c))
                continue;

            int v = getCell(s, r, c);
            if (seen[v])
                return false;
            seen[v] = 1;
        }
    }

    for (int c = 0; c < s.n; ++c)
    {
        std::vector<int> seen(s.n, 0);
        for (int r = 0; r < s.n; ++r)
        {
            if (!isAssigned(s, r, c))
                continue;

            int v = getCell(s, r, c);
            if (seen[v])
                return false;
            seen[v] = 1;
        }
    }

    // 3. Optional stronger propagation: fill forced entries.
    //    This can catch contradictions earlier and reduce branching.
    //    For example:
    //    if table[a][b] is known and a*a must be a, then we can infer required values.

    return true;
}
```

---

## 4. Candidate selection and branching

```cpp
Cell chooseNextUnassignedCell(const State& s)
{
    Cell best = {-1, -1};
    int bestScore = INT_MAX;

    for (int i = 0; i < s.n; ++i)
    {
        for (int j = 0; j < s.n; ++j)
        {
            if (isAssigned(s, i, j))
                continue;

            int count = 0;
            for (int v = 0; v < s.n; ++v)
            {
                State child = s;
                setCell(child, i, j, v);
                if (isFeasible(child))
                    ++count;
            }

            if (count < bestScore)
            {
                bestScore = count;
                best = {i, j};
            }
        }
    }

    return best;
}

std::vector<int> generateCandidates(const State& s, const Cell& cell)
{
    std::vector<int> out;

    for (int v = 0; v < s.n; ++v)
    {
        State child = s;
        setCell(child, cell.row, cell.col, v);
        if (isFeasible(child))
            out.push_back(v);
    }

    return out;
}
```

---

## 5. State extension and assignment

```cpp
State applyAssignment(const State& s, const Cell& cell, int value)
{
    State child = s;
    setCell(child, cell.row, cell.col, value);
    child.depth += 1;
    child.assignedCount = child.depth;
    child.complete = (child.depth == s.n * s.n);
    child.valid = true;
    return child;
}
```

---

## 6. Completion and validity checks

```cpp
bool isComplete(const State& s)
{
    return s.assignedCount == s.n * s.n;
}

bool isValidQuandle(const State& s)
{
    if (!isComplete(s))
        return false;

    for (int a = 0; a < s.n; ++a)
    {
        for (int b = 0; b < s.n; ++b)
        {
            if (!axiom1Holds(s, a, b))
                return false;

            if (!axiom2Holds(s, a, b))
                return false;
        }
    }

    // Quandle check for all rows and columns being permutations
    for (int a = 0; a < s.n; ++a)
    {
        std::vector<int> rowSeen(s.n, 0);
        std::vector<int> colSeen(s.n, 0);

        for (int b = 0; b < s.n; ++b)
        {
            int rowVal = getCell(s, a, b);
            int colVal = getCell(s, b, a);

            if (rowSeen[rowVal] || colSeen[colVal])
                return false;

            rowSeen[rowVal] = 1;
            colSeen[colVal] = 1;
        }
    }

    return true;
}
```

---

## 7. CUDA kernel for batched pruning

```cpp
__global__ void pruneAndExpandKernel(
    const State* inStates,
    int inCount,
    Result* outResults,
    int* outCount)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= inCount)
        return;

    State s = inStates[tid];

    if (!deviceIsFeasible(s))
    {
        outResults[tid].kind = Result::INVALID;
        outResults[tid].state = s;
        return;
    }

    if (isCompleteDevice(s))
    {
        if (deviceIsValidQuandle(s))
        {
            outResults[tid].kind = Result::COMPLETE_VALID;
            outResults[tid].state = s;
        }
        else
        {
            outResults[tid].kind = Result::INVALID;
            outResults[tid].state = s;
        }
        return;
    }

    Cell cell = chooseNextUnassignedCellDevice(s);
    int childCount = 0;
    State children[16];

    for (int value = 0; value < s.n; ++value)
    {
        State child = applyAssignmentDevice(s, cell, value);
        if (deviceIsFeasible(child))
        {
            children[childCount++] = child;
        }
    }

    if (childCount == 0)
    {
        outResults[tid].kind = Result::INVALID;
        outResults[tid].state = s;
        return;
    }

    // Store a compact list of child states in a preallocated output buffer.
    // A fixed-size per-thread output region avoids dynamic memory allocation.
    int base = tid * 16;
    for (int i = 0; i < childCount; ++i)
    {
        outResults[base + i].kind = Result::PARTIAL_VALID;
        outResults[base + i].state = children[i];
    }

    // Optional: write a small count for each thread.
    outCount[tid] = childCount;
}
```

Important implementation detail:

- GPU threads should not recursively expand arbitrary subtrees.
- The GPU should only do bounded local pruning or one-level branching.
- Returning too many children per thread can overflow the output buffer.
- A safe design is to cap the child count per thread and let the CPU continue deeper recursion when necessary.

---

## 8. Device feasibility check

```cpp
__device__ bool deviceIsFeasible(const State& s)
{
    for (int i = 0; i < s.n; ++i)
    {
        for (int j = 0; j < s.n; ++j)
        {
            if (!isAssignedDevice(s, i, j))
                continue;

            if (!deviceAxiom1(s, i, j))
                return false;

            if (!deviceAxiom2(s, i, j))
                return false;

            if (!deviceAxiom3Partial(s, i, j))
                return false;
        }
    }

    for (int r = 0; r < s.n; ++r)
    {
        bool seen[256];
        memset(seen, 0, sizeof(seen));
        for (int c = 0; c < s.n; ++c)
        {
            if (!isAssignedDevice(s, r, c))
                continue;

            int v = getCellDevice(s, r, c);
            if (seen[v])
                return false;
            seen[v] = true;
        }
    }

    for (int c = 0; c < s.n; ++c)
    {
        bool seen[256];
        memset(seen, 0, sizeof(seen));
        for (int r = 0; r < s.n; ++r)
        {
            if (!isAssignedDevice(s, r, c))
                continue;

            int v = getCellDevice(s, r, c);
            if (seen[v])
                return false;
            seen[v] = true;
        }
    }

    return true;
}
```

---

## 9. Full CUDA entrypoint

```cpp
std::vector<Result> runGpuBatch(const std::vector<State>& batch)
{
    int count = (int)batch.size();

    State* d_states = nullptr;
    Result* d_results = nullptr;
    int* d_counts = nullptr;

    cudaMalloc(&d_states, count * sizeof(State));
    cudaMalloc(&d_results, count * 16 * sizeof(Result));
    cudaMalloc(&d_counts, count * sizeof(int));

    cudaMemcpy(d_states, batch.data(), count * sizeof(State), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocks = (count + threadsPerBlock - 1) / threadsPerBlock;

    pruneAndExpandKernel<<<blocks, threadsPerBlock>>>(
        d_states,
        count,
        d_results,
        d_counts
    );

    cudaDeviceSynchronize();

    std::vector<Result> hostResults(count * 16);
    cudaMemcpy(hostResults.data(), d_results, count * 16 * sizeof(Result), cudaMemcpyDeviceToHost);

    std::vector<Result> filtered;
    for (int i = 0; i < count * 16; ++i)
    {
        if (hostResults[i].kind == Result::INVALID)
            continue;

        filtered.push_back(hostResults[i]);
    }

    cudaFree(d_states);
    cudaFree(d_results);
    cudaFree(d_counts);

    return filtered;
}
```

This is the point where the CPU and GPU are coordinated:

- the CPU decides when to offload work,
- the GPU validates a batch and expands only feasible partial states,
- the CPU re-enters backtracking with the returned valid states.

---

## 10. Important design constraints

### A. Avoid recursive calls inside CUDA kernels
A CUDA kernel should not recursively call a full search routine. Instead, it should do:

- feasibility filtering,
- one-step branching,
- a bounded number of child outputs.

Otherwise the GPU will diverge badly and blow up memory.

### B. Use fixed-size output slots
This is the right way to avoid serialization issues:

- each thread reserves a fixed array of slots,
- each slot holds a result object,
- the CPU reads the count and only processes those slots.

### C. Prefer pruning over expansion
The goal of the GPU is not to fully solve the tree. It is to eliminate impossible states before they ever reach the CPU.

### D. Canonicalize completed states
For isomorphism reduction, complete valid quandles should be canonicalized before insertion into the seen set.

---

## 11. Final combined pseudocode summary

```cpp
void explore()
{
    std::vector<State> frontier;
    std::unordered_set<std::string> completed;

    initFrontier(frontier);

    while (!frontier.empty())
    {
        if (frontier.size() > GPU_TRIGGER)
        {
            std::vector<State> batch = takeBatch(frontier, GPU_BATCH_SIZE);
            std::vector<Result> results = runGpuBatch(batch);

            for (const Result& r : results)
            {
                if (r.kind == Result::INVALID)
                    continue;

                if (r.kind == Result::COMPLETE_VALID)
                {
                    std::string key = canonicalize(r.state);
                    if (!completed.count(key))
                    {
                        completed.insert(key);
                        record(r.state);
                    }
                }
                else if (r.kind == Result::PARTIAL_VALID)
                {
                    frontier.push_back(r.state);
                }
            }
        }
        else
        {
            State s = frontier.back();
            frontier.pop_back();

            if (!isFeasible(s))
                continue;

            if (isComplete(s))
            {
                if (isValidQuandle(s))
                {
                    std::string key = canonicalize(s);
                    if (!completed.count(key))
                    {
                        completed.insert(key);
                        record(s);
                    }
                }
                continue;
            }

            Cell cell = chooseNextUnassignedCell(s);
            for (int value : generateCandidates(s, cell))
            {
                State child = applyAssignment(s, cell, value);
                if (isFeasible(child))
                    frontier.push_back(child);
            }
        }
    }
}
```

This is the clean hybrid strategy: the CPU performs recursive backtracking, and the GPU performs batched pruning and bounded expansion to reduce the search space.
