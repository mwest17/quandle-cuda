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

constexpr int ORDER = 3;
constexpr int TABLE_SIZE = ORDER * ORDER;
constexpr int MAX_CHILDREN_PER_THREAD = ORDER;

typedef int8_t element;
typedef int16_t index;

struct Cell
{
    element row = -1;
    element col = -1;
};

struct Orbit
{
    index size = 0;
    element orb[ORDER] = {-1};
};

template <size_t N>
struct state_t
{
    // TODO** Optimize this representation for memory
    index assignedCount = 0;
    element table[N * N] = {-1}; // Bitpack the structure maybe? Only if I run out of memory
    Cell lastEntered = {-1, -1};
};

typedef state_t<ORDER> State;

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



void printOrbits(const std::vector<Orbit>& orbits);
void printQuandle(const State& s);


__host__ __device__ inline index stateIndex(element row, element col)
{
    return row * ORDER + col;
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
    for (element i = 0; i < ORDER; i++)
    {
        if (op(s, i, y) == z)
        {
            x = i;
            break;
        }
    }
    return x;
}   

inline bool isComplete(const State& s);

std::vector<Orbit> determineOrbits(const State& s)
{
    if (!isComplete(s))
        return {};

    for (element row = 0; row < ORDER; ++row)
    {
        for (element col = 0; col < ORDER; ++col)
        {
            const element value = static_cast<element>(op(s, row, col));
            if (value < 0 || value >= ORDER)
                return {};
        }
    }

    bool assigned[ORDER] = {};
    std::vector<Orbit> orbits;

    for (element start = 0; start < ORDER; ++start)
    {
        if (assigned[start])
            continue;

        Orbit orbit;
        element queue[ORDER] = {start};
        bool visited[ORDER] = {};
        int head = 0;
        int tail = 1;
        visited[start] = true;

        while (head < tail)
        {
            const element current = queue[head++];
            orbit.orb[orbit.size++] = current;
            assigned[current] = true;

            for (element actingElement = 0; actingElement < ORDER; ++actingElement)
            {
                const element next = static_cast<element>(op(s, current, actingElement));
                const element previous = static_cast<element>(op_inv(s, current, actingElement));

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

        orbits.push_back(orbit);
    }

    return orbits;
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
    for (element x = 0; x < ORDER; x++)
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
    for (element a = 0; a < ORDER; a++)
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
    for (element a = 0; a < ORDER; a++)
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
    for (element a = 0; a < ORDER; a++)
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
    for (element a = 0; a < ORDER; a++)
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
    for (element a = 0; a < ORDER; a++)
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

    for (element row = 0; row < ORDER; ++row)
    {
        for (element col = 0; col < ORDER; ++col)
        {
            const element firstValue = op(first, row, col);
            const element secondValue = op(second, row, col);
            if (firstValue < 0 || firstValue >= ORDER || secondValue < 0 || secondValue >= ORDER)
                return false;
        }
    }

    using Signature = std::array<int, ORDER + 1>;
    auto translationSignature = [](const State& state, element elementValue, bool right) {
        Signature signature{};
        bool imageSeen[ORDER] = {};

        for (element input = 0; input < ORDER; ++input)
        {
            const element image = right
                ? static_cast<element>(op(state, input, elementValue))
                : static_cast<element>(op(state, elementValue, input));
            if (image < 0 || image >= ORDER || imageSeen[image])
                return Signature{};
            imageSeen[image] = true;
        }

        bool visited[ORDER] = {};
        for (element input = 0; input < ORDER; ++input)
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
        bool visited[ORDER] = {};
        element queue[ORDER] = {start};
        int head = 0;
        int tail = 1;
        visited[start] = true;

        while (head < tail)
        {
            const element current = queue[head++];
            for (element actingElement = 0; actingElement < ORDER; ++actingElement)
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

    Signature firstRowSignatures[ORDER]{};
    Signature secondRowSignatures[ORDER]{};
    Signature firstColumnSignatures[ORDER]{};
    Signature secondColumnSignatures[ORDER]{};
    int firstOrbitSizes[ORDER]{};
    int secondOrbitSizes[ORDER]{};

    for (element value = 0; value < ORDER; ++value)
    {
        firstRowSignatures[value] = translationSignature(first, value, false);
        secondRowSignatures[value] = translationSignature(second, value, false);
        firstColumnSignatures[value] = translationSignature(first, value, true);
        secondColumnSignatures[value] = translationSignature(second, value, true);
        firstOrbitSizes[value] = orbitSize(first, value);
        secondOrbitSizes[value] = orbitSize(second, value);
    }

    auto sameMultiset = [](const auto& firstValues, const auto& secondValues) {
        bool matched[ORDER] = {};
        for (element firstIndex = 0; firstIndex < ORDER; ++firstIndex)
        {
            bool found = false;
            for (element secondIndex = 0; secondIndex < ORDER; ++secondIndex)
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

    int mapping[ORDER];
    int inverseMapping[ORDER];
    for (element value = 0; value < ORDER; ++value)
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

        for (element other = 0; other < ORDER; ++other)
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
        if (mappedCount == ORDER)
            return true;

        element source = -1;
        int bestCandidateCount = ORDER + 1;
        for (element candidateSource = 0; candidateSource < ORDER; ++candidateSource)
        {
            if (mapping[candidateSource] != -1)
                continue;

            int candidateCount = 0;
            for (element candidateTarget = 0; candidateTarget < ORDER; ++candidateTarget)
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

        for (element target = 0; target < ORDER; ++target)
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
    std::vector<Orbit> orbits = determineOrbits(s);

    // Cohen quandles must have at least 2 orbits
    if (orbits.size() < 2)
        return false;

    // All orbits must have the same size to be isomorphic
    const index firstOrbitSize = orbits[0].size;
    for (int i = 1; i < orbits.size(); ++i)
    {
        if (orbits[i].size != firstOrbitSize)
            return false;
    }

    const index SUB_QUANDLE_SIZE = ORDER - firstOrbitSize;
    std::array<state_t<SUB_QUANDLE_SIZE>, orbits.size()> orbitStates;

    for (int i = 0; i < orbits.size(); ++i)
    {
        const Orbit& orbit = orbits[i];
        state_t<SUB_QUANDLE_SIZE>& orbitSubquandle = orbitStates[i];

        orbitSubquandle.assignedCount = SUB_QUANDLE_SIZE;

        for (int j = 0; j < ORDER; j++)
        {
            if 
        }

        // for (int j = 0; j < firstOrbitSize; ++j)
        // {
        //     element row = orbit.orb[j];
        //     for (element col = 0; col < ORDER; ++col)
        //     {
        //         if (std::find(std::begin(orbit.orb), std::end(orbit.orb), col) == std::end(orbit.orb))
        //         {
        //             setCell(orbitState, j, col, op(s, row, col));
        //         }
        //     }
        // }
    }
    // Check if s / orb is isomoprhism to all others
    // Just do the half matrix pyramid approach to compare

}
#endif

inline bool isValidQuandle(const State& s)
{
    // for (int x = 0; x < ORDER; ++x)
    // {
    //     if (op(s, x, x) != x)
    //         return false;
    // }

    // for (int x = 0; x < ORDER; x++)
    // {
    //     for (int y = 0; y < ORDER; y++)
    //     {
    //         if (op_inv(s, op(s, x, y), y) != x)
    //             return false;
    //     }
    // }

    for (element x = 0; x < ORDER; x++)
    {
        for (element y = 0; y < ORDER; y++)
        {
            for (element z = 0; z < ORDER; z++)
            {
                if (op(s, op(s, x, y), z) != op(s, op(s, x, z), op(s, y, z)))
                    return false;
            }
        }
    }

    // for (int r = 0; r < ORDER; ++r)
    // {
    //     int rowSeen[ORDER] = {};
    //     for (int c = 0; c < ORDER; ++c)
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

    for (element row = 0; row < ORDER; ++row)
    {
        for (element col = 0; col < ORDER; ++col)
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
    out.reserve(ORDER);
    for (element value = 0; value < ORDER; ++value)
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
//     for (int value = 0; value < ORDER && childCount < MAX_CHILDREN_PER_THREAD; ++value)
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

    for (element i = 0; i < ORDER; ++i)
    {
        for (element j = 0; j < ORDER; ++j)
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
        for (element row = 0; row < ORDER; ++row)
        {
            for (element col = 0; col < ORDER; ++col)
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

void testDetermineOrbits()
{
    std::cout << "Running determineOrbits test 1...\n";
    State s;
    s.assignedCount = 9;
    s.table[0] = 0; s.table[1] = 1; s.table[2] = 2;
    s.table[3] = 1; s.table[4] = 2; s.table[5] = 0;
    s.table[6] = 2; s.table[7] = 0; s.table[8] = 1;

    printQuandle(s);
    std::vector<Orbit> orbits = determineOrbits(s);
    std::cout << "Number of orbits: " << orbits.size() << std::endl;
    printOrbits(orbits);

    std::cout << "\nRunning determineOrbits test 2...\n";
    s.table[0] = 0; s.table[1] = 0; s.table[2] = 0;
    s.table[3] = 1; s.table[4] = 1; s.table[5] = 1;
    s.table[6] = 2; s.table[7] = 2; s.table[8] = 2;
    printQuandle(s);
    orbits = determineOrbits(s);
    std::cout << "Number of orbits: " << orbits.size() << std::endl;
    printOrbits(orbits);

    std::cout << "\nRunning determineOrbits test 3...\n";
    s.table[0] = 0; s.table[1] = 0; s.table[2] = 1;
    s.table[3] = 1; s.table[4] = 1; s.table[5] = 0;
    s.table[6] = 2; s.table[7] = 2; s.table[8] = 2;
    printQuandle(s);
    orbits = determineOrbits(s);
    std::cout << "Number of orbits: " << orbits.size() << std::endl;
    printOrbits(orbits);
}

void printOrbits(const std::vector<Orbit>& orbits)
{
    std::cout << "Orbits: ";
    for (size_t i = 0; i < orbits.size(); ++i)
    {
        const Orbit& orbit = orbits[i];
        std::cout << "{";
        for (element j = 0; j < orbit.size; ++j)
        {
            std::cout << static_cast<int>(orbit.orb[j]);
            std::cout << ((j < orbit.size - 1) ? ", " : "");
        }
        std::cout << "} ";
    }
    std::cout << std::endl;
}

void printQuandle(const State& s)
{
    for (element row = 0; row < ORDER; ++row)
    {
        for (element col = 0; col < ORDER; ++col)
        {
            std::cout << static_cast<int>(op(s, row, col) + 1) << ' ';
        }
        std::cout << '\n';
    }
}

int main()
{
    // isomorphismTest();
    // testDetermineOrbits();

    std::cout << "Running hybrid CPU/GPU quandle backtracking search\n";
    searchHybrid();

    return 0;
}