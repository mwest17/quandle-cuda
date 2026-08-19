#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <limits>
#include <string>
#include <unordered_set>
#include <vector>

constexpr int N = 3;
constexpr int TABLE_SIZE = N * N;
constexpr int MAX_CHILDREN_PER_THREAD = N;

struct Cell
{
    int row = -1;
    int col = -1;
};

struct State
{
    // TODO** Optimize this representation for memory
    int assignedCount = 0;
    int table[TABLE_SIZE] = {-1}; // Bitpack the structure maybe?
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

__host__ __device__ inline int stateIndex(int row, int col)
{
    return row * N + col;
}

__host__ __device__ inline bool isAssigned(const State& s, int row, int col)
{
    return s.table[stateIndex(row, col)] != -1;
}

__host__ __device__ inline int op(const State& s, int row, int col)
{
    return s.table[stateIndex(row, col)];
}

__host__ __device__ inline int op_inv(const State& s, int z, int y)
{
    int x = -1;
    for (int i = 0; i < N; i++)
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
    const int idx = stateIndex(row, col);
    s.table[idx] = value;
    s.assignedCount += 1;
    s.lastEntered = {row, col};
}

__host__ __device__ inline bool verifyPartialAxiomTwo(const State& s, short row, short col, short k)
{
    for (int x = 0; x < N; x++)
    {
        if (op(s, x, col) == k && x != row)
        {
            return false;
        }
    }

    // TODO** If only 1 left in the column, fill in with only remaining value
    return true;
}

__host__ __device__ inline bool ruleOne(State& s, short row, short col, short k)
{
    for (short a = 0; a < N; a++)
    {
        short j_a = op(s, row, a);
        short i_a = op(s, col, a);

        // Rule 1: k * a = (j * a) * (i * a)
        if (j_a != -1 && i_a != -1)
        {
            std::cout << "j_a: " << j_a << " i_a: " << i_a << std::endl;
            
            short j_a_i_a = op(s, j_a, i_a);
            short k_a = op(s, k, a);

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

__host__ __device__ inline bool ruleTwo(State& s, short row, short col, short k)
{
    for (short a = 0; a < N; a++)
    {
        short a_j = op(s, a, row);
        short a_i = op(s, a, col);

        if (a_j != -1 && a_i != -1)
        {
            short a_j_i = op(s, a_j, col);
            short a_i_k = op(s, a_i, k);
            
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

__host__ __device__ inline bool ruleThree(State& s, short row, short col, short k)
{
    for (short a = 0; a < N; a++)
    {
        short j_a = op(s, row, a);
        short a_i = op(s, a, col);

        if (j_a != -1 && a_i != -1)
        {
            short j_a_i = op(s, j_a, col);
            short k_a_i = op(s, k, a_i);

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

__host__ __device__ inline bool ruleFour(State& s, short row, short col, short k)
{
    for (short a = 0; a < N; a++)
    {
        short j_inv_a = op_inv(s, row, a);
        short i_inv_a = op_inv(s, col, a);

        if (j_inv_a != -1 && i_inv_a != -1)
        {
            short j_inv_a_i_inv_a = op(s, j_inv_a, i_inv_a);

            if (j_inv_a_i_inv_a != -1)
            {
                short j_inv_a_i_inv_a_a = op(s, j_inv_a_i_inv_a, a);

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

__host__ __device__ inline bool ruleFive(State& s, short row, short col, short k)
{
    for (short a = 0; a < N; a++)
    {
        short j_inv_a = op_inv(s, row, a);
        short a_i = op(s, a, col);

        if (j_inv_a != -1 && a_i != -1)
        {
            short j_inv_a_i = op(s, j_inv_a, col);

            if (j_inv_a_i != -1)
            {
                short j_inv_a_i_a_i = op(s, j_inv_a_i, a_i);

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
    short row = s.lastEntered.row;
    short col = s.lastEntered.col;
    short k = op(s, row, col);

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

    return result;
}

bool isComplete(const State& s)
{
    return s.assignedCount == TABLE_SIZE;
}

bool isValidQuandle(const State& s)
{
    for (int x = 0; x < N; x++)
    {
        for (int y = 0; y < N; y++)
        {
            for (int z = 0; z < N; z++)
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

    for (int row = 0; row < N; ++row)
    {
        for (int col = 0; col < N; ++col)
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
    for (int value = 0; value < N; ++value)
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

    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < N; ++j)
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
                    completeStates.push_back(s);
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

    std::cout << "completed quandle states found: " << completeStates.size() << '\n';
    for (size_t i = 0; i < completeStates.size(); ++i)
    {
        const State& q = completeStates[i];
        std::cout << "State " << i << ":" << '\n';
        for (int row = 0; row < N; ++row)
        {
            for (int col = 0; col < N; ++col)
            {
                std::cout << q.table[stateIndex(row, col)] << ' ';
            }
            std::cout << '\n';
        }
        std::cout << '\n';
    }
    std::cout << "Total states explored: " << statesExplored << "\n";
}

int main()
{
    std::cout << "Running hybrid CPU/GPU quandle backtracking search\n";

    auto start = std::chrono::high_resolution_clock::now();

    searchHybrid();
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Elapsed time: " << elapsed.count() << " seconds\n";

    return 0;
}