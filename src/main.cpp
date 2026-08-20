#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <limits>
#include <string>
#include <unordered_set>
#include <vector>

constexpr int N = 8;
constexpr int TABLE_SIZE = N * N;
constexpr int MAX_CHILDREN_PER_THREAD = N;

typedef int8_t element;
typedef int16_t index;

struct Cell
{
    element row = -1;
    element col = -1;
};

struct Orbit
{
    element orb[N] = {-1};
};

struct State
{
    // TODO** Optimize this representation for memory
    index assignedCount = 0;
    element table[TABLE_SIZE] = {-1}; // Bitpack the structure maybe?
    Cell lastEntered = {-1, -1};
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



void printQuandle(const State& s);


__host__ __device__ inline index stateIndex(element row, element col)
{
    return row * N + col;
}

__host__ __device__ inline bool isAssigned(const State& s, element row, element col)
{
    return s.table[stateIndex(row, col)] != -1;
}

__host__ __device__ inline index op(const State& s, element row, element col)
{
    return s.table[stateIndex(row, col)];
}

__host__ __device__ inline index op_inv(const State& s, element z, element y)
{
    element x = -1;
    for (element i = 0; i < N; i++)
    {
        if (op(s, i, y) == z)
        {
            x = i;
            break;
        }
    }
    return x;
}   

__host__ __device__ inline void setCell(State& s, element row, element col, element value)
{
    const index idx = stateIndex(row, col);
    s.table[idx] = value;
    s.assignedCount += 1;
    s.lastEntered = {row, col};
}

__host__ __device__ inline bool verifyPartialAxiomTwo(const State& s, element row, element col, element k)
{
    for (element x = 0; x < N; x++)
    {
        if (op(s, x, col) == k && x != row)
        {
            return false;
        }
    }

    // TODO** If only 1 left in the column, fill in with only remaining value
    return true;
}

__host__ __device__ inline bool ruleOne(State& s, element row, element col, element k)
{
    for (element a = 0; a < N; a++)
    {
        element j_a = op(s, row, a);
        element i_a = op(s, col, a);

        // Rule 1: k * a = (j * a) * (i * a)
        if (j_a != -1 && i_a != -1)
        {   
            element j_a_i_a = op(s, j_a, i_a);
            element k_a = op(s, k, a);

            // (j * a) * (i * a) cannot be retrieved and k * a can be retrieved
            if (j_a_i_a == -1 && k_a != -1) // 1
            {
                if (op_inv(s, k_a, i_a) != -1) // a
                    return false; // (k * a) *^-1 (i * a) can be retrieved
                else // b
                    setCell(s, j_a, i_a, k_a);
            }
            // (j * a) * (i * a) can be retrieved and k * a cannot be retrieved 
            else if (j_a_i_a != -1 && k_a == -1) // 2
            {
                if (op_inv(s, j_a_i_a, a) != -1) // a
                    return false;
                else // b
                    setCell(s, k, a, j_a_i_a);
            } 
            else if (j_a_i_a != -1 && k_a != -1) // 3
            {
                if (j_a_i_a != k_a)
                    return false;
            }
        }
    }

    return true;
}

__host__ __device__ inline bool ruleTwo(State& s, element row, element col, element k)
{
    for (element a = 0; a < N; a++)
    {
        element a_j = op(s, a, row);
        element a_i = op(s, a, col);

        if (a_j != -1 && a_i != -1)
        {
            element a_j_i = op(s, a_j, col);
            element a_i_k = op(s, a_i, k);
            
            if (a_i_k == -1 && a_j_i != -1) // 1
            {
                if (op_inv(s, a_j_i, k) != -1) // a
                    return false;
                else // b
                    setCell(s, a_i, k, a_j_i);
            }
            else if (a_j_i == -1 && a_i_k != -1) // 2
            {
                if (op_inv(s, a_i_k, col) != -1) // a
                    return false;
                else // b
                    setCell(s, a_j, col, a_i_k);
            } 
            else if (a_j_i != -1 && a_i_k != 1) // 3
            {
                if (a_j_i != a_i_k)
                    return false;
            }
        }
    }
    return true;
}

__host__ __device__ inline bool ruleThree(State& s, element row, element col, element k)
{
    for (element a = 0; a < N; a++)
    {
        element j_a = op(s, row, a);
        element a_i = op(s, a, col);

        if (j_a != -1 && a_i != -1)
        {
            element j_a_i = op(s, j_a, col);
            element k_a_i = op(s, k, a_i);

            if (k_a_i == -1 && j_a_i != -1) // 1
            {
                if (op_inv(s, j_a_i, a_i) != -1) // a
                    return false;
                else // b
                    setCell(s, k, a_i, j_a_i);
            } 
            else if (j_a_i == -1 && k_a_i != -1) // 2
            {
                if (op_inv(s, k_a_i, col) != -1)
                    return false;
                else
                    setCell(s, j_a, col, k_a_i);
            }
            else if (j_a_i != -1 && k_a_i != -1) // 3
            {
                if (j_a_i != k_a_i)
                    return false;
            }
        }
    }
    return true;
}

__host__ __device__ inline bool ruleFour(State& s, element row, element col, element k)
{
    for (element a = 0; a < N; a++)
    {
        element j_inv_a = op_inv(s, row, a);
        element i_inv_a = op_inv(s, col, a);

        if (j_inv_a != -1 && i_inv_a != -1)
        {
            element j_inv_a_i_inv_a = op(s, j_inv_a, i_inv_a);

            if (j_inv_a_i_inv_a != -1)
            {
                element j_inv_a_i_inv_a_a = op(s, j_inv_a_i_inv_a, a);

                if (j_inv_a_i_inv_a_a == -1) // 1
                {
                    if (op_inv(s, k, a) != -1) // a
                        return false;
                    else
                        setCell(s, j_inv_a_i_inv_a, a, k); // b
                }
                else // 2
                {
                    if (j_inv_a_i_inv_a_a != k)
                        return false;
                }
            }
        }
    }
    return true;
}

__host__ __device__ inline bool ruleFive(State& s, element row, element col, element k)
{
    for (element a = 0; a < N; a++)
    {
        element j_inv_a = op_inv(s, row, a);
        element a_i = op(s, a, col);

        if (j_inv_a != -1 && a_i != -1)
        {
            element j_inv_a_i = op(s, j_inv_a, col);

            if (j_inv_a_i != -1)
            {
                element j_inv_a_i_a_i = op(s, j_inv_a_i, a_i);

                if (j_inv_a_i_a_i == -1) // 1
                {
                    if (op_inv(s, k, a_i) != -1) // a
                        return false;
                    else // b
                        setCell(s, j_inv_a_i, a_i, k);
                }
                else // 2
                {
                    if (k != j_inv_a_i_a_i)
                        return false;
                }
            }
        }
    }
    return true;
}

// Side Effects: Will fill in any cells that are a consequence of the new cell
__host__ __device__ inline bool isFeasible(State& s)
// TODO** Try to avoid divergence??? Somehow?? CUDA programmers have nightmares about this algorithm

// Ensure that no rules dictated by quandle axioms are broken
// Ensure that no orbits are expanded or broken
// OPTIONAL: Ensure that cohen conditions are still met
{
    element row = s.lastEntered.row;
    element col = s.lastEntered.col;
    element k = op(s, row, col);

    bool result = verifyPartialAxiomTwo(s, row, col, k);

    if (result)
        result = ruleOne(s, row, col, k);
    
    if (result)
        result = ruleTwo(s, row, col, k);

    if (result)
        result = ruleThree(s, row, col, k); 

    if (result)
        result = ruleFour(s, row, col, k);

    if (result)
        result = ruleFive(s, row, col, k);

#ifdef COHEN
    if (result)
        result = cohenParitalCheck();
#endif

    return result;
}

inline bool isComplete(const State& s)
{
    return s.assignedCount == TABLE_SIZE;
}

bool areIsomorphic(const State& first, const State& second)
{
    if (!isComplete(first) || !isComplete(second))
        return false;

    for (element row = 0; row < N; ++row)
    {
        for (element col = 0; col < N; ++col)
        {
            const element firstValue = op(first, row, col);
            const element secondValue = op(second, row, col);
            if (firstValue < 0 || firstValue >= N || secondValue < 0 || secondValue >= N)
                return false;
        }
    }

    using Signature = std::array<int, N + 1>;
    auto translationSignature = [](const State& state, element elementValue, bool right) {
        Signature signature{};
        bool imageSeen[N] = {};

        for (element input = 0; input < N; ++input)
        {
            const element image = right
                ? static_cast<element>(op(state, input, elementValue))
                : static_cast<element>(op(state, elementValue, input));
            if (image < 0 || image >= N || imageSeen[image])
                return Signature{};
            imageSeen[image] = true;
        }

        bool visited[N] = {};
        for (element input = 0; input < N; ++input)
        {
            if (visited[input])
                continue;

            int cycleLength = 0;
            element current = input;
            do
            {
                visited[current] = true;
                current = right
                    ? static_cast<element>(op(state, current, elementValue))
                    : static_cast<element>(op(state, elementValue, current));
                ++cycleLength;
            } while (current != input);
            ++signature[cycleLength];
        }
        return signature;
    };

    auto orbitSize = [](const State& state, element start) {
        bool visited[N] = {};
        element queue[N] = {start};
        int head = 0;
        int tail = 1;
        visited[start] = true;

        while (head < tail)
        {
            const element current = queue[head++];
            for (element actingElement = 0; actingElement < N; ++actingElement)
            {
                const element next = static_cast<element>(op(state, current, actingElement));
                const element previous = op_inv(state, current, actingElement);
                if (!visited[next])
                {
                    visited[next] = true;
                    queue[tail++] = next;
                }
                if (previous >= 0 && !visited[previous])
                {
                    visited[previous] = true;
                    queue[tail++] = previous;
                }
            }
        }
        return tail;
    };

    Signature firstRowSignatures[N]{};
    Signature secondRowSignatures[N]{};
    Signature firstColumnSignatures[N]{};
    Signature secondColumnSignatures[N]{};
    int firstOrbitSizes[N]{};
    int secondOrbitSizes[N]{};

    for (element value = 0; value < N; ++value)
    {
        firstRowSignatures[value] = translationSignature(first, value, false);
        secondRowSignatures[value] = translationSignature(second, value, false);
        firstColumnSignatures[value] = translationSignature(first, value, true);
        secondColumnSignatures[value] = translationSignature(second, value, true);
        firstOrbitSizes[value] = orbitSize(first, value);
        secondOrbitSizes[value] = orbitSize(second, value);
    }

    auto sameMultiset = [](const auto& firstValues, const auto& secondValues) {
        bool matched[N] = {};
        for (element firstIndex = 0; firstIndex < N; ++firstIndex)
        {
            bool found = false;
            for (element secondIndex = 0; secondIndex < N; ++secondIndex)
            {
                if (!matched[secondIndex] && firstValues[firstIndex] == secondValues[secondIndex])
                {
                    matched[secondIndex] = true;
                    found = true;
                    break;
                }
            }
            if (!found)
                return false;
        }
        return true;
    };

    if (!sameMultiset(firstRowSignatures, secondRowSignatures) ||
        !sameMultiset(firstColumnSignatures, secondColumnSignatures) ||
        !sameMultiset(firstOrbitSizes, secondOrbitSizes))
    {
        return false;
    }

    int mapping[N];
    int inverseMapping[N];
    for (element value = 0; value < N; ++value)
    {
        mapping[value] = -1;
        inverseMapping[value] = -1;
    }

    auto compatible = [&](element source, element target) {
        if (firstRowSignatures[source] != secondRowSignatures[target] ||
            firstColumnSignatures[source] != secondColumnSignatures[target] ||
            firstOrbitSizes[source] != secondOrbitSizes[target])
        {
            return false;
        }

        for (element other = 0; other < N; ++other)
        {
            if (mapping[other] == -1)
                continue;

            const element sourceResults[2] = {
                static_cast<element>(op(first, source, other)),
                static_cast<element>(op(first, other, source))};
            const element targetResults[2] = {
                static_cast<element>(op(second, target, mapping[other])),
                static_cast<element>(op(second, mapping[other], target))};

            for (int direction = 0; direction < 2; ++direction)
            {
                const element sourceResult = sourceResults[direction];
                const element targetResult = targetResults[direction];
                if (mapping[sourceResult] != -1 && mapping[sourceResult] != targetResult)
                    return false;
                if (inverseMapping[targetResult] != -1 && inverseMapping[targetResult] != sourceResult)
                    return false;
            }
        }
        return true;
    };

    auto search = [&](auto&& self, int mappedCount) -> bool {
        if (mappedCount == N)
            return true;

        element source = -1;
        int bestCandidateCount = N + 1;
        for (element candidateSource = 0; candidateSource < N; ++candidateSource)
        {
            if (mapping[candidateSource] != -1)
                continue;

            int candidateCount = 0;
            for (element candidateTarget = 0; candidateTarget < N; ++candidateTarget)
            {
                if (inverseMapping[candidateTarget] == -1 && compatible(candidateSource, candidateTarget))
                    ++candidateCount;
            }
            if (candidateCount < bestCandidateCount)
            {
                source = candidateSource;
                bestCandidateCount = candidateCount;
            }
        }

        if (source == -1 || bestCandidateCount == 0)
            return false;

        for (element target = 0; target < N; ++target)
        {
            if (inverseMapping[target] != -1 || !compatible(source, target))
                continue;

            mapping[source] = target;
            inverseMapping[target] = source;
            if (self(self, mappedCount + 1))
                return true;
            mapping[source] = -1;
            inverseMapping[target] = -1;
        }
        return false;
    };

    return search(search, 0);
}

#ifdef COHEN
inline bool isCohen(const State& s)
{
    // Get all orbits


    // Check if s / orb is isomoprhism to all others
    // Just do the half matrix pyramid approach to compare
}
#endif

inline bool isValidQuandle(const State& s)
{
    // for (int x = 0; x < N; ++x)
    // {
    //     if (op(s, x, x) != x)
    //         return false;
    // }

    // for (int x = 0; x < N; x++)
    // {
    //     for (int y = 0; y < N; y++)
    //     {
    //         if (op_inv(s, op(s, x, y), y) != x)
    //             return false;
    //     }
    // }

    for (element x = 0; x < N; x++)
    {
        for (element y = 0; y < N; y++)
        {
            for (element z = 0; z < N; z++)
            {
                if (op(s, op(s, x, y), z) != op(s, op(s, x, z), op(s, y, z)))
                    return false;
            }
        }
    }

    // for (int r = 0; r < N; ++r)
    // {
    //     int rowSeen[N] = {};
    //     for (int c = 0; c < N; ++c)
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
    // TODO** Choose cell that will lead to most results the quickest
    // TODO** Possibly store the previous filled in cell for quicker selection
    Cell next{-1, -1};

    for (element row = 0; row < N; ++row)
    {
        for (element col = 0; col < N; ++col)
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
    out.reserve(N);
    for (element value = 0; value < N; ++value)
    {
        State child = s;
        setCell(child, cell.row, cell.col, value);
        if (isFeasible(child))
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
//     if (!isFeasible(s))
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
//     for (int value = 0; value < N && childCount < MAX_CHILDREN_PER_THREAD; ++value)
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

std::vector<State> buildInitialRoots()
{
    std::vector<State> roots;
    State s;
    s.assignedCount = 0;

    for (element i = 0; i < N; ++i)
    {
        for (element j = 0; j < N; ++j)
        {
            if (i == j)
            {
                s.table[stateIndex(i, j)] = i;
                s.assignedCount++;
            }
            else
            {
                s.table[stateIndex(i, j)] = -1;
            }
        }
    }

    roots.push_back(s);
    return roots;
}

void searchHybrid()
{
    auto start = std::chrono::high_resolution_clock::now();

    constexpr int GPU_TRIGGER = 1024;
    constexpr int GPU_BATCH_SIZE = 1024;

    long long statesExplored = 0;

    std::vector<State> stack = buildInitialRoots();
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
        //             completeStates.push_back(r.state);
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
            statesExplored++;

            if (isComplete(s))
            {
                if (isValidQuandle(s))
                {
                    bool isDuplicate = false;
                    for (const State& existing : completeStates)
                    {
                        if (areIsomorphic(existing, s))
                        {
                            isDuplicate = true;
                            break;
                        }
                    }

                    if (!isDuplicate)
                    {
                        completeStates.push_back(s);
                    }
                }
                continue;
            }

            const Cell cell = chooseNextUnassignedCell(s);
            const std::vector<State> candidates = generateCandidates(s, cell);
            for (const State& child : candidates)
            {
                stack.push_back(child);
            }
        // }
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    for (size_t i = 0; i < completeStates.size(); ++i)
    {
        const State& q = completeStates[i];
        std::cout << "State " << i << ":" << '\n';
        for (element row = 0; row < N; ++row)
        {
            for (element col = 0; col < N; ++col)
            {
                std::cout << static_cast<int>(q.table[stateIndex(row, col)]) << ' ';
            }
            std::cout << '\n';
        }
        std::cout << '\n';
    }
    std::cout << "Total states explored: " << statesExplored << "\n";
    std::cout << "Quandles found: " << completeStates.size() << '\n';
    std::cout << "Elapsed time: " << elapsed.count() << " seconds\n";
}

void isomorphismTest()
{
    std::cout << "Running isomorphism test 1...\n";
    State s1;
    s1.assignedCount = 9;
    s1.table[0] = 0; s1.table[1] = 1; s1.table[2] = 2;
    s1.table[3] = 1; s1.table[4] = 2; s1.table[5] = 0;
    s1.table[6] = 2; s1.table[7] = 0; s1.table[8] = 1;

    State s2;
    s2.assignedCount = 9;
    s2.table[0] = 0; s2.table[1] = 2; s2.table[2] = 1;
    s2.table[3] = 2; s2.table[4] = 1; s2.table[5] = 0;
    s2.table[6] = 1; s2.table[7] = 0; s2.table[8] = 2;

    printQuandle(s1);
    std::cout << std::endl;
    printQuandle(s2);

    bool isomorphic = areIsomorphic(s1, s2);
    std::cout << ((isomorphic)? "Isomorphic" : "Not Isomorphic") << std::endl;
    std::cout << "Test 1 isomorphic: " << ((isomorphic == false)? "Pass" : "Fail") << std::endl;


    std::cout << "Running isomorphism test 2...\n";
    s1.table[0] = 0; s1.table[1] = 1; s1.table[2] = 2;
    s1.table[3] = 1; s1.table[4] = 2; s1.table[5] = 0;
    s1.table[6] = 2; s1.table[7] = 0; s1.table[8] = 1;

    s2.table[0] = 0; s2.table[1] = 1; s2.table[2] = 2;
    s2.table[3] = 1; s2.table[4] = 2; s2.table[5] = 0;
    s2.table[6] = 2; s2.table[7] = 0; s2.table[8] = 1;

    printQuandle(s1);
    std::cout << std::endl;
    printQuandle(s2);

    isomorphic = areIsomorphic(s1, s2);
    std::cout << ((isomorphic)? "Isomorphic" : "Not Isomorphic") << std::endl;
    std::cout << "Test 2 isomorphic: " << ((isomorphic == true)? "Pass" : "Fail") << std::endl;

    
    std::cout << "Running isomorphism test 3...\n";
    s1.table[0] = 0; s1.table[1] = 0; s1.table[2] = 1;
    s1.table[3] = 1; s1.table[4] = 1; s1.table[5] = 0;
    s1.table[6] = 2; s1.table[7] = 2; s1.table[8] = 2;

    s2.table[0] = 0; s2.table[1] = 0; s2.table[2] = 0;
    s2.table[3] = 2; s2.table[4] = 1; s2.table[5] = 1;
    s2.table[6] = 1; s2.table[7] = 2; s2.table[8] = 2;

    printQuandle(s1);
    std::cout << std::endl;
    printQuandle(s2);

    isomorphic = areIsomorphic(s1, s2);
    std::cout << ((isomorphic)? "Isomorphic" : "Not Isomorphic") << std::endl;
    std::cout << "Test 3 isomorphic: " << ((isomorphic == true)? "Pass" : "Fail") << std::endl;
}

void printQuandle(const State& s)
{
    for (element row = 0; row < N; ++row)
    {
        for (element col = 0; col < N; ++col)
        {
            std::cout << static_cast<int>(op(s, row, col)) << ' ';
        }
        std::cout << '\n';
    }
}

int main()
{
    // isomorphismTest();
    std::cout << "Running hybrid CPU/GPU quandle backtracking search\n";

    searchHybrid();

    return 0;
}