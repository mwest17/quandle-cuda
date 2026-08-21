#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

constexpr int ORDER = 12;
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
#ifdef COHEN
    index orbitSize = 0;
#endif
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
void printQuandle(const std::vector<element>& quandle);
void stateToVector(const State& s, std::vector<element>& vec);
inline bool isComplete(const State& s);


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

#ifdef COHEN
    if ((k / s.orbitSize) != (row / s.orbitSize))
        return false;

    // Can also ensure a row isn't about to become trivial
    if (k == row && row == ORDER - 1)
    {
        bool trivial = true;
        for (int i = 0; i < ORDER; i++)
        {
            if (op(s, row, i) != k)
            {
                trivial = false;
                break;
            }
        }

        if (trivial)
            return false;
    }
#endif

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

inline bool isComplete(const State& s)
{
    return s.assignedCount == TABLE_SIZE;
}

bool areIsomorphic(const std::vector<element>& first, const std::vector<element>& second)
{
    if (first.size() != second.size())
        return false;

    const size_t tableSize = first.size();
    size_t order = 0;
    while (order * order < tableSize)
        ++order;

    if (order * order != tableSize)
        return false;

    for (size_t value = 0; value < tableSize; ++value)
    {
        if (first[value] < 0 || first[value] >= order ||
            second[value] < 0 || second[value] >= order)
        {
            return false;
        }
    }

    auto translationSignature = [order](const std::vector<element>& table, size_t value, bool right) {
        std::vector<int> signature(order + 1);
        bool imageSeen[ORDER] = {};
        for (size_t input = 0; input < order; ++input)
        {
            const element image = right
                ? table[input * order + value]
                : table[value * order + input];
            if (image < 0 || image >= order || imageSeen[image])
                return signature;
            imageSeen[image] = true;
        }

        bool visited[ORDER] = {};
        for (size_t input = 0; input < order; ++input)
        {
            if (visited[input])
                continue;

            int cycleLength = 0;
            size_t current = input;
            do
            {
                visited[current] = true;
                current = right
                    ? table[current * order + value]
                    : table[value * order + current];
                ++cycleLength;
            } while (current != input);
            ++signature[cycleLength];
        }
        return signature;
    };

    auto orbitSize = [order](const std::vector<element>& table, size_t start) {
        bool visited[ORDER] = {};
        size_t queue[ORDER] = {start};
        size_t head = 0;
        size_t tail = 1;
        visited[start] = true;

        while (head < tail)
        {
            const size_t current = queue[head++];
            for (size_t actingElement = 0; actingElement < order; ++actingElement)
            {
                const size_t next = table[current * order + actingElement];
                size_t previous = 0;
                while (table[previous * order + actingElement] != current)
                    ++previous;

                if (!visited[next])
                {
                    visited[next] = true;
                    queue[tail++] = next;
                }
                if (!visited[previous])
                {
                    visited[previous] = true;
                    queue[tail++] = previous;
                }
            }
        }
        return tail;
    };

    std::vector<std::vector<int>> firstRowSignatures(order);
    std::vector<std::vector<int>> secondRowSignatures(order);
    std::vector<std::vector<int>> firstColumnSignatures(order);
    std::vector<std::vector<int>> secondColumnSignatures(order);
    std::vector<size_t> firstOrbitSizes(order);
    std::vector<size_t> secondOrbitSizes(order);

    for (size_t value = 0; value < order; ++value)
    {
        firstRowSignatures[value] = translationSignature(first, value, false);
        secondRowSignatures[value] = translationSignature(second, value, false);
        firstColumnSignatures[value] = translationSignature(first, value, true);
        secondColumnSignatures[value] = translationSignature(second, value, true);
        firstOrbitSizes[value] = orbitSize(first, value);
        secondOrbitSizes[value] = orbitSize(second, value);
    }

    auto sameMultiset = [](const auto& firstValues, const auto& secondValues) {
        std::vector<bool> matched(secondValues.size());
        for (size_t firstIndex = 0; firstIndex < firstValues.size(); ++firstIndex)
        {
            bool found = false;
            for (size_t secondIndex = 0; secondIndex < secondValues.size(); ++secondIndex)
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

    std::vector<element> mapping(order);
    for (element value = 0; value < order; ++value)
        mapping[value] = value;

    do
    {
        bool isCompatible = true;
        for (element row = 0; row < order && isCompatible; ++row)
        {
            for (element col = 0; col < order; ++col)
            {
                const element firstValue = first[row * order + col];
                const element mappedValue = mapping[firstValue];
                const element secondValue = second[mapping[row] * order + mapping[col]];
                if (mappedValue != secondValue)
                {
                    isCompatible = false;
                    break;
                }
            }
        }

        if (isCompatible)
            return true;
    } while (std::next_permutation(mapping.begin(), mapping.end()));

    return false;
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

    const index quotientSize = ORDER - firstOrbitSize;
    using QuotientTable = std::vector<element>;
    std::vector<QuotientTable> quotientTables;
    quotientTables.reserve(orbits.size());

    for (const Orbit& orbit : orbits)
    {
        bool removed[ORDER] = {};
        for (element orbitIndex = 0; orbitIndex < orbit.size; ++orbitIndex)
            removed[orbit.orb[orbitIndex]] = true;

        element oldToNew[ORDER];
        for (element value = 0; value < ORDER; ++value)
            oldToNew[value] = -1;

        element nextIndex = 0;
        for (element value = 0; value < ORDER; ++value)
        {
            if (!removed[value])
                oldToNew[value] = nextIndex++;
        }

        QuotientTable quotient(static_cast<size_t>(quotientSize) * quotientSize, -1);
        for (element row = 0; row < ORDER; ++row)
        {
            if (removed[row])
                continue;

            for (element col = 0; col < ORDER; ++col)
            {
                if (removed[col])
                    continue;

                const element value = static_cast<element>(op(s, row, col));
                if (value < 0 || value >= ORDER || removed[value])
                    return false;

                const element newRow = oldToNew[row];
                const element newCol = oldToNew[col];
                const element newValue = oldToNew[value];
                if (newRow < 0 || newCol < 0 || newValue < 0)
                    return false;

                quotient[static_cast<size_t>(newRow) * quotientSize + newCol] = newValue;
            }
        }

        quotientTables.push_back(std::move(quotient));
    }

    for (int i = 0; i < quotientTables.size(); ++i)
    {
        for (int j = i + 1; j < quotientTables.size(); ++j)
        {
            if (!areIsomorphic(quotientTables.at(i), quotientTables.at(j)))
                return false;
        }
    }

    return true;
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

std::vector<std::vector<element>> parseQuandleFile(std::string filename)
{
    std::vector<std::vector<element>> quandleData;

    std::ifstream file(filename);
    if (!file.is_open())
    {
        std::cout << "Error opening file: " << filename << std::endl;
        return quandleData;
    }

    std::vector<element> quandle;
    std::string line;
    while (std::getline(file, line))
    {
        if (line.empty())
        {
            quandleData.push_back(quandle);
            quandle.clear();
        }
        else if (line[0] == '[' && line[1] != '{')
        {
            std::istringstream iss(line.substr(1, line.find(']') - 1));
            int tmp;
            // element value;
            while (iss >> tmp)
            {
                quandle.push_back(tmp - 1);
            }
        }
    }
    return quandleData;
}

std::vector<std::vector<element>> readQuandlesFromFile(int n)
{
    // First get cohen of order n
    std::vector<std::vector<element>> quandles = parseQuandleFile("../cohen/cohen" + std::to_string(n) + ".txt");

    // Then get connected of order n
    std::vector<std::vector<element>> connectedQuandles = parseQuandleFile("../connected/connected" + std::to_string(n) + ".txt");

    // Join the two vectors
    quandles.insert(quandles.end(), connectedQuandles.begin(), connectedQuandles.end());

    return quandles;
}

std::vector<State> buildInitialRoots()
{
    std::vector<State> roots;

    std::vector<int> possibleOrbitSizes;

#ifdef COHEN
    State trivial;
    trivial.assignedCount = TABLE_SIZE;
    trivial.orbitSize = 1;
    for (int i = 0; i < ORDER; i++) // Add trivial to roots as it is always cohen and would be skipped otherwise
    {
        for (int j = 0; j < ORDER; j++)
        {
            trivial.table[stateIndex(i, j)] = i;
        }
    }
    roots.push_back(std::move(trivial));

    for (int size = 2; size < ORDER; ++size)
    {
        if (ORDER % size == 0)
            possibleOrbitSizes.push_back(size);
    }

#ifdef TEST
    std::cout << "Possible orbit sizes: ";
    for (int size : possibleOrbitSizes)
    {
        std::cout << size << " "; 
    }
    std::cout << std::endl;
#endif

    for (int orbitSize : possibleOrbitSizes)
    {
        // read quandles of that order from file
        std::vector<std::vector<element>> quandleData = readQuandlesFromFile(orbitSize);

#ifdef TEST
        std::cout << "Orbit Size: " << orbitSize << std::endl;
        std::cout << "Number of possible orbits: " << quandleData.size() << std::endl;
#endif

        for (std::vector<element> quandle : quandleData)
        {
            if (quandle.size() < 1)
                continue;

            const index numOrbits = ORDER / orbitSize;
            State s;
            s.assignedCount = (orbitSize * orbitSize) * (numOrbits);
            s.orbitSize = orbitSize;
            std::fill_n(s.table, TABLE_SIZE, -1);

#ifdef TEST
            printQuandle(quandle);
            std::cout << "Num Orbits: " << numOrbits << std::endl;
#endif

            // insert each quandle as the orbit of a state
            for (int i = 0; i < numOrbits; i++)
            {
                // insert the quandle as the orbit of the state
                index offset = i * orbitSize;

                for (element row = 0; row < orbitSize; row++)
                {
                    for (element col = 0; col < orbitSize; col++)
                    {
                        // std::cout << "(" << static_cast<int>(row) << ", " << static_cast<int>(col) << ") " << static_cast<int>(quandle[row * orbitSize + col]) << std::endl;
                        s.table[stateIndex(row + offset, col + offset)] = quandle[row * orbitSize + col] + offset;
                    }
                }
            }
            roots.push_back(std::move(s));
        }
    }
#else
    // TODO** Update to fill in orbits like the Cohen init
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
#endif

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
                    #ifdef COHEN
                    if (!isCohen(s))
                        continue;
                    #endif
                    
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
        std::cout << "State " << i << ":" << '\n';
        printQuandle(completeStates[i]);
        std::cout << '\n';
    }
    std::cout << "Total states explored: " << statesExplored << "\n";
    std::cout << "Quandles found: " << completeStates.size() << '\n';
    std::cout << "Elapsed time: " << elapsed.count() << " seconds\n";
}

void stateToVector(const State& s, std::vector<element>& vec)
{
    vec.resize(TABLE_SIZE);
    for (element row = 0; row < ORDER; ++row)
    {
        for (element col = 0; col < ORDER; ++col)
        {
            vec[stateIndex(row, col)] = s.table[stateIndex(row, col)];
        }
    }
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

    std::vector<element> s1Vec;
    std::vector<element> s2Vec;

    printQuandle(s1);
    std::cout << std::endl;
    printQuandle(s2);

    bool isomorphic = areIsomorphic(s1, s2);
    std::cout << ((isomorphic)? "Isomorphic" : "Not Isomorphic") << std::endl;
    std::cout << "Test 1 isomorphic: " << ((isomorphic == false)? "Pass" : "Fail") << std::endl;

    stateToVector(s1, s1Vec);
    stateToVector(s2, s2Vec);

    bool isomorphicVec = areIsomorphic(s1Vec, s2Vec);
    std::cout << ((isomorphicVec)? "Isomorphic" : "Not Isomorphic") << std::endl;
    std::cout << "Test 1 isomorphic (vector): " << ((isomorphicVec == false)? "Pass" : "Fail") << std::endl;


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

    stateToVector(s1, s1Vec);
    stateToVector(s2, s2Vec);
    
    isomorphicVec = areIsomorphic(s1Vec, s2Vec);
    std::cout << ((isomorphicVec)? "Isomorphic" : "Not Isomorphic") << std::endl;
    std::cout << "Test 2 isomorphic (vector): " << ((isomorphicVec == true)? "Pass" : "Fail") << std::endl;

    
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

    stateToVector(s1, s1Vec);
    stateToVector(s2, s2Vec);
    
    isomorphicVec = areIsomorphic(s1Vec, s2Vec);
    std::cout << ((isomorphicVec)? "Isomorphic" : "Not Isomorphic") << std::endl;
    std::cout << "Test 3 isomorphic (vector): " << ((isomorphicVec == true)? "Pass" : "Fail") << std::endl;
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

#ifdef COHEN
void testIsCohen()
{
    std::cout << "Running isCohen test 1...\n";
    State s;
    s.assignedCount = 9;
    s.table[0] = 0; s.table[1] = 1; s.table[2] = 2;
    s.table[3] = 1; s.table[4] = 2; s.table[5] = 0;
    s.table[6] = 2; s.table[7] = 0; s.table[8] = 1;

    printQuandle(s);
    bool cohen = isCohen(s);
    std::cout << "Is Cohen: " << (cohen ? "Yes" : "No") << std::endl;

    std::cout << "\nRunning isCohen test 2...\n";
    s.table[0] = 0; s.table[1] = 0; s.table[2] = 0;
    s.table[3] = 1; s.table[4] = 1; s.table[5] = 1;
    s.table[6] = 2; s.table[7] = 2; s.table[8] = 2;
    printQuandle(s);
    cohen = isCohen(s);
    std::cout << "Is Cohen: " << (cohen ? "Yes" : "No") << std::endl;

    std::cout << "\nRunning isCohen test 3...\n";
    s.table[0] = 0; s.table[1] = 0; s.table[2] = 1;
    s.table[3] = 1; s.table[4] = 1; s.table[5] = 0;
    s.table[6] = 2; s.table[7] = 2; s.table[8] = 2;
    printQuandle(s);
    cohen = isCohen(s);
    std::cout << "Is Cohen: " << (cohen ? "Yes" : "No") << std::endl;
}
#endif

void testReadFromFile()
{
    std::vector<std::vector<element>> quandles = readQuandlesFromFile(ORDER);

    for (const std::vector<element>& quandle : quandles)
    {
        printQuandle(quandle);
        std::cout << std::endl;
    }
}

void testBuildInitialRoots()
{
    std::vector<State> roots = buildInitialRoots();

    for (const State& root : roots)
    {
        std::cout << "Root state:\n";
        printQuandle(root);
        std::cout << std::endl;
    }
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

void printQuandle(const std::vector<element>& quandle)
{
    int n = std::sqrt(quandle.size());
    for (element row = 0; row < n; ++row)
    {
        for (element col = 0; col < n; ++col)
        {
            std::cout << quandle[row * n + col] + 1 << ' ';
        }
        std::cout << '\n';
    }
}

int main()
{
#ifdef TEST
    // isomorphismTest();
    // testDetermineOrbits();
    // testIsCohen();
    // testReadFromFile();
    testBuildInitialRoots();
#else
    std::cout << "Running hybrid CPU/GPU quandle backtracking search\n";
    searchHybrid();
#endif

    return 0;
}