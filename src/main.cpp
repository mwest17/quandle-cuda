#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <limits>
#include <string>
#include <unordered_set>
#include <vector>

constexpr int MAX_ORDER = 8;
constexpr int MAX_TABLE_SIZE = MAX_ORDER * MAX_ORDER;
constexpr int MAX_CHILDREN_PER_THREAD = 16;

struct Cell
{
    int row = -1;
    int col = -1;
};

struct State
{
    int n = 0;
    int assignedCount = 0;
    int table[MAX_TABLE_SIZE] = {};
    bool valid = true;
    bool complete = false;
};

struct Result
{
    enum Kind
    {
        INVALID,
        PARTIAL_VALID,
        COMPLETE_VALID
    };

    Kind kind = INVALID;
    State state{};
};

__host__ __device__ inline int stateIndex(int row, int col, int n)
{
    return row * n + col;
}

__host__ __device__ inline bool isAssigned(const State& s, int row, int col)
{
    return s.table[stateIndex(row, col, s.n)] != -1;
}

__host__ __device__ inline int op(const State& s, int row, int col)
{
    return s.table[stateIndex(row, col, s.n)];
}

__host__ __device__ inline int op_inv(const State& s, int z, int y)
{
    int x = -1;
    for (int i = 0; i < s.n; i++)
    {
        if (op(s, i, y) == z)
        {
            x = i;
            break;
        }
    }
    return x;
}   

__host__ __device__ inline void setCell(State& s, int row, int col, int value)
{
    const int idx = stateIndex(row, col, s.n);
    s.table[idx] = value;
    s.assignedCount += 1;
    s.complete = (s.assignedCount == s.n * s.n);
}

__host__ __device__ inline bool deviceIsFeasible(const State& s)
{
    for (int i = 0; i < s.n; ++i)
    {
        for (int j = 0; j < s.n; ++j)
        {
            if (!isAssigned(s, i, j))
                continue;
        }
    }

    for (int r = 0; r < s.n; ++r)
    {
        int seen[MAX_ORDER] = {};
        for (int c = 0; c < s.n; ++c)
        {
            if (!isAssigned(s, r, c))
                continue;

            int value = op(s, r, c);
            if (value < 0 || value >= s.n)
                return false;
            if (seen[value])
                return false;
            seen[value] = 1;
        }
    }

    for (int c = 0; c < s.n; ++c)
    {
        int seen[MAX_ORDER] = {};
        for (int r = 0; r < s.n; ++r)
        {
            if (!isAssigned(s, r, c))
                continue;

            int value = op(s, r, c);
            if (value < 0 || value >= s.n)
                return false;
            if (seen[value])
                return false;
            seen[value] = 1;
        }
    }

    return true;
}

inline bool isFeasible(const State& s)
{
    return deviceIsFeasible(s);
}

bool isComplete(const State& s)
{
    return s.assignedCount == s.n * s.n;
}

bool isValidQuandle(const State& s)
{
    for (int x = 0; x < s.n; ++x)
    {
        if (op(s, x, x) != x)
            return false;
    }

    for (int x = 0; x < s.n; x++)
    {
        for (int y = 0; y < s.n; y++)
        {
            if (op_inv(s, op(s, x, y), y) != x)
                return false;
        }
    }

    for (int x = 0; x < s.n; x++)
    {
        for (int y = 0; y < s.n; y++)
        {
            for (int z = 0; z < s.n; z++)
            {
                if (op(s, op(s, x, y), z) != op(s, op(s, x, z), op(s, y, z)))
                    return false;
            }
        }
    }

    // for (int r = 0; r < s.n; ++r)
    // {
    //     int rowSeen[MAX_ORDER] = {};
    //     for (int c = 0; c < s.n; ++c)
    //     {
    //         const int value = op(s, r, c);
    //         if (rowSeen[value])
    //             return false;
    //         rowSeen[value] = 1;
    //     }
    // }

    return true;
}

Cell chooseNextUnassignedCell(const State& s)
{
    Cell next{-1, -1};

    for (int row = 0; row < s.n; ++row)
    {
        for (int col = 0; col < s.n; ++col)
        {
            if (isAssigned(s, row, col))
                continue;

            next = {row, col};
            break;
        }
    }

    return next;
}

std::vector<State> generateCandidates(const State& s, const Cell& cell)
{
    std::vector<State> out;
    for (int value = 0; value < s.n; ++value)
    {
        State child = s;
        setCell(child, cell.row, cell.col, value);
        // if (isFeasible(child))
        out.push_back(child);
    }
    return out;
}

// __global__ void pruneAndExpandKernel(
//     const State* inStates,
//     int inCount,
//     Result* outResults,
//     int* outCounts)
// {
//     const int tid = blockIdx.x * blockDim.x + threadIdx.x;
//     if (tid >= inCount)
//         return;
//
//     const State s = inStates[tid];
//     if (!deviceIsFeasible(s))
//     {
//         outResults[tid * MAX_CHILDREN_PER_THREAD].kind = Result::INVALID;
//         outResults[tid * MAX_CHILDREN_PER_THREAD].state = s;
//         outCounts[tid] = 0;
//         return;
//     }
//
//     if (isComplete(s))
//     {
//         if (isValidQuandle(s))
//         {
//             outResults[tid * MAX_CHILDREN_PER_THREAD].kind = Result::COMPLETE_VALID;
//             outResults[tid * MAX_CHILDREN_PER_THREAD].state = s;
//         }
//         else
//         {
//             outResults[tid * MAX_CHILDREN_PER_THREAD].kind = Result::INVALID;
//             outResults[tid * MAX_CHILDREN_PER_THREAD].state = s;
//         }
//         outCounts[tid] = 1;
//         return;
//     }
//
//     Cell cell = chooseNextUnassignedCell(s);
//     State children[MAX_CHILDREN_PER_THREAD];
//     int childCount = 0;
//
//     for (int value = 0; value < s.n && childCount < MAX_CHILDREN_PER_THREAD; ++value)
//     {
//         State child = s;
//         setCell(child, cell.row, cell.col, value);
//         if (isFeasible(child))
//         {
//             children[childCount++] = child;
//         }
//     }
//
//     if (childCount == 0)
//     {
//         outResults[tid * MAX_CHILDREN_PER_THREAD].kind = Result::INVALID;
//         outResults[tid * MAX_CHILDREN_PER_THREAD].state = s;
//         outCounts[tid] = 0;
//         return;
//     }
//
//     for (int i = 0; i < childCount; ++i)
//     {
//         outResults[tid * MAX_CHILDREN_PER_THREAD + i].kind = Result::PARTIAL_VALID;
//         outResults[tid * MAX_CHILDREN_PER_THREAD + i].state = children[i];
//     }
//
//     for (int i = childCount; i < MAX_CHILDREN_PER_THREAD; ++i)
//     {
//         outResults[tid * MAX_CHILDREN_PER_THREAD + i].kind = Result::INVALID;
//     }
//
//     outCounts[tid] = childCount;
// }
//
// std::vector<Result> runGpuBatch(const std::vector<State>& batch)
// {
//     if (batch.empty())
//         return {};
//
//     State* d_states = nullptr;
//     Result* d_results = nullptr;
//     int* d_counts = nullptr;
//
//     const size_t stateBytes = batch.size() * sizeof(State);
//     const size_t resultBytes = batch.size() * MAX_CHILDREN_PER_THREAD * sizeof(Result);
//     const size_t countBytes = batch.size() * sizeof(int);
//
//     cudaMalloc(&d_states, stateBytes);
//     cudaMalloc(&d_results, resultBytes);
//     cudaMalloc(&d_counts, countBytes);
//
//     cudaMemcpy(d_states, batch.data(), stateBytes, cudaMemcpyHostToDevice);
//
//     const int threadsPerBlock = 256;
//     const int blocks = static_cast<int>((batch.size() + threadsPerBlock - 1) / threadsPerBlock);
//
//     pruneAndExpandKernel<<<blocks, threadsPerBlock>>>(d_states, static_cast<int>(batch.size()), d_results, d_counts);
//     cudaDeviceSynchronize();
//
//     std::vector<Result> hostResults(batch.size() * MAX_CHILDREN_PER_THREAD);
//     cudaMemcpy(hostResults.data(), d_results, resultBytes, cudaMemcpyDeviceToHost);
//
//     std::vector<int> counts(batch.size());
//     cudaMemcpy(counts.data(), d_counts, countBytes, cudaMemcpyDeviceToHost);
//
//     std::vector<Result> filtered;
//     for (size_t i = 0; i < batch.size(); ++i)
//     {
//         for (int j = 0; j < counts[i]; ++j)
//         {
//             const Result r = hostResults[i * MAX_CHILDREN_PER_THREAD + j];
//             if (r.kind != Result::INVALID)
//                 filtered.push_back(r);
//         }
//     }
//
//     cudaFree(d_states);
//     cudaFree(d_results);
//     cudaFree(d_counts);
//
//     return filtered;
// }

std::vector<State> buildInitialRoots(int n)
{
    std::vector<State> roots;
    State s;
    s.n = n;
    s.assignedCount = 0;
    s.valid = true;
    s.complete = false;

    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            s.table[stateIndex(i, j, n)] = -1;
        }
    }

    roots.push_back(s);
    return roots;
}

void searchHybrid(int n)
{
    constexpr int GPU_TRIGGER = 1024;
    constexpr int GPU_BATCH_SIZE = 1024;

    std::vector<State> stack = buildInitialRoots(n);
    std::unordered_set<std::string> seen;
    std::vector<State> completeStates;

    while (!stack.empty())
    {
        // if (static_cast<int>(stack.size()) > GPU_TRIGGER) // **TODO Only do CPU right now
        // {
        //     std::vector<State> batch;
        //     for (int i = 0; i < GPU_BATCH_SIZE && !stack.empty(); ++i)
        //     {
        //         batch.push_back(stack.back());
        //         stack.pop_back();
        //     }

        //     const auto results = runGpuBatch(batch);
        //     for (const Result& r : results)
        //     {
        //         if (r.kind == Result::INVALID)
        //             continue;

        //         if (r.kind == Result::COMPLETE_VALID)
        //         {
        //             const std::string key = canonicalize(r.state);
        //             if (seen.insert(key).second)
        //             {
        //                 completeStates.push_back(r.state);
        //             }
        //             continue;
        //         }

        //         if (r.kind == Result::PARTIAL_VALID)
        //         {
        //             stack.push_back(r.state);
        //         }
        //     }
        // }
        // else
        // {
            State s = stack.back();
            stack.pop_back();

            if (isComplete(s))
            {
                if (isValidQuandle(s))
                {
                    completeStates.push_back(s);
                }
                continue;
            }

            const Cell cell = chooseNextUnassignedCell(s);
            const std::vector<State> candidates = generateCandidates(s, cell);
            for (State child : candidates)
            {
                stack.push_back(child);
            }
        // }
    }

    std::cout << "completed quandle states found: " << completeStates.size() << '\n';
    for (size_t i = 0; i < completeStates.size(); ++i)
    {
        const State& q = completeStates[i];
        std::cout << "State " << i << ":" << '\n';
        for (int row = 0; row < q.n; ++row)
        {
            for (int col = 0; col < q.n; ++col)
            {
                std::cout << q.table[stateIndex(row, col, q.n)] << ' ';
            }
            std::cout << '\n';
        }
        std::cout << '\n';
    }
}

int main()
{
    std::cout << "Running hybrid CPU/GPU quandle backtracking search\n";

    int order = 3;
    searchHybrid(order);

    return 0;
}