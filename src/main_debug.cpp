#include <algorithm>
#include <cstdint>
#include <iostream>
#include <limits>
#include <string>
#include <unordered_set>
#include <vector>

constexpr int MAX_ORDER = 8;
constexpr int MAX_TABLE_SIZE = MAX_ORDER * MAX_ORDER;

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
    int assigned[MAX_TABLE_SIZE] = {};
    bool valid = true;
    bool complete = false;
};

inline int stateIndex(int row, int col, int n)
{
    return row * n + col;
}

inline bool isAssigned(const State& s, int row, int col)
{
    return s.assigned[stateIndex(row, col, s.n)] != 0;
}

inline int getCell(const State& s, int row, int col)
{
    return s.table[stateIndex(row, col, s.n)];
}

inline void setCell(State& s, int row, int col, int value)
{
    const int idx = stateIndex(row, col, s.n);
    s.table[idx] = value;
    s.assigned[idx] = 1;
    s.assignedCount += 1;
    s.complete = (s.assignedCount == s.n * s.n);
}

inline bool isFeasible(const State& s)
{
    // 1. Check idempotence: if (a, a) is assigned, it must equal a
    for (int a = 0; a < s.n; ++a)
    {
        if (isAssigned(s, a, a) && getCell(s, a, a) != a)
            return false;
    }

    // 2. Check the second axiom: if (a * b) and ((a * b) * b) are both assigned,
    //    then ((a * b) * b) must equal a
    for (int a = 0; a < s.n; ++a)
    {
        for (int b = 0; b < s.n; ++b)
        {
            if (!isAssigned(s, a, b))
                continue;

            const int ab = getCell(s, a, b);
            if (ab < 0 || ab >= s.n)
                return false;

            if (isAssigned(s, ab, b))
            {
                if (getCell(s, ab, b) != a)
                    return false;
            }
        }
    }

    // 3. Check row uniqueness
    for (int r = 0; r < s.n; ++r)
    {
        int seen[MAX_ORDER] = {};
        for (int c = 0; c < s.n; ++c)
        {
            if (!isAssigned(s, r, c))
                continue;

            int value = getCell(s, r, c);
            if (value < 0 || value >= s.n)
                return false;
            if (seen[value])
                return false;
            seen[value] = 1;
        }
    }

    // 4. Check column uniqueness
    for (int c = 0; c < s.n; ++c)
    {
        int seen[MAX_ORDER] = {};
        for (int r = 0; r < s.n; ++r)
        {
            if (!isAssigned(s, r, c))
                continue;

            int value = getCell(s, r, c);
            if (value < 0 || value >= s.n)
                return false;
            if (seen[value])
                return false;
            seen[value] = 1;
        }
    }

    return true;
}

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
        if (getCell(s, a, a) != a)
            return false;

        for (int b = 0; b < s.n; ++b)
        {
            const int x = getCell(s, a, b);
            if (x < 0 || x >= s.n)
                return false;

            if (getCell(s, x, b) != a)
                return false;

            if (getCell(s, b, b) != b)
                return false;
        }
    }

    for (int r = 0; r < s.n; ++r)
    {
        int rowSeen[MAX_ORDER] = {};
        for (int c = 0; c < s.n; ++c)
        {
            const int value = getCell(s, r, c);
            if (rowSeen[value])
                return false;
            rowSeen[value] = 1;
        }
    }

    for (int c = 0; c < s.n; ++c)
    {
        int colSeen[MAX_ORDER] = {};
        for (int r = 0; r < s.n; ++r)
        {
            const int value = getCell(s, r, c);
            if (colSeen[value])
                return false;
            colSeen[value] = 1;
        }
    }

    return true;
}

Cell chooseNextUnassignedCell(const State& s)
{
    for (int row = 0; row < s.n; ++row)
    {
        for (int col = 0; col < s.n; ++col)
        {
            if (!isAssigned(s, row, col))
                return {row, col};
        }
    }
    return {-1, -1};
}

std::vector<int> generateCandidates(const State& s, const Cell& cell)
{
    std::vector<int> out;
    for (int value = 0; value < s.n; ++value)
    {
        State child = s;
        setCell(child, cell.row, cell.col, value);
        if (isFeasible(child))
            out.push_back(value);
    }
    return out;
}

State applyAssignment(const State& s, const Cell& cell, int value)
{
    State child = s;
    setCell(child, cell.row, cell.col, value);
    child.valid = true;
    child.complete = (child.assignedCount == s.n * s.n);
    return child;
}

std::string canonicalize(const State& s)
{
    std::string key;
    for (int i = 0; i < s.n * s.n; ++i)
    {
        key.push_back(static_cast<char>('0' + s.table[i]));
    }
    return key;
}

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
            s.assigned[stateIndex(i, j, n)] = 0;
        }
    }

    roots.push_back(s);
    return roots;
}

void printState(const State& s, int nodeNum)
{
    std::cout << "Node " << nodeNum << ": assigned=" << s.assignedCount << "/" << (s.n * s.n) << "\n";
    for (int r = 0; r < s.n; ++r)
    {
        for (int c = 0; c < s.n; ++c)
        {
            if (isAssigned(s, r, c))
                std::cout << (char)('0' + getCell(s, r, c));
            else
                std::cout << ".";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

void searchCPU(int n)
{
    std::vector<State> stack = buildInitialRoots(n);
    std::unordered_set<std::string> seen;
    std::vector<State> completeStates;
    int nodesExplored = 0;
    int totalComplete = 0;

    while (!stack.empty())
    {
        State s = stack.back();
        stack.pop_back();
        nodesExplored++;

        if (!isFeasible(s))
        {
            if (nodesExplored <= 50)
                std::cout << "Node " << nodesExplored << ": PRUNED (infeasible)\n";
            continue;
        }

        if (nodesExplored <= 50)
            printState(s, nodesExplored);

        if (isComplete(s))
        {
            if (isValidQuandle(s))
            {
                totalComplete++;
                const std::string key = canonicalize(s);
                std::cout << "** FOUND COMPLETE QUANDLE #" << totalComplete << " **\n";
                if (seen.insert(key).second)
                {
                    completeStates.push_back(s);
                }
                else
                {
                    std::cout << "  (duplicate - already seen this one)\n";
                }
            }
            else
            {
                std::cout << "Node " << nodesExplored << ": COMPLETE but INVALID\n";
            }
            continue;
        }

        const Cell cell = chooseNextUnassignedCell(s);
        if (cell.row == -1)
            continue;

        const std::vector<int> candidates = generateCandidates(s, cell);
        if (candidates.empty())
        {
            if (nodesExplored <= 50)
                std::cout << "Node " << nodesExplored << ": PRUNED (no candidates for (" << cell.row << "," << cell.col << "))\n";
        }
        else if (nodesExplored <= 50)
        {
            std::cout << "Node " << nodesExplored << ": Cell (" << cell.row << "," << cell.col << ") has " << candidates.size() << " candidates: ";
            for (int v : candidates) std::cout << v << " ";
            std::cout << "\n";
        }
        
        for (int value : candidates)
        {
            State child = applyAssignment(s, cell, value);
            if (isFeasible(child))
                stack.push_back(child);
        }
    }

    std::cout << "\n========================================\n";
    std::cout << "CPU Backtracking Search Results (n=" << n << ")\n";
    std::cout << "========================================\n";
    std::cout << "Nodes explored: " << nodesExplored << '\n';
    std::cout << "Total complete valid states found: " << totalComplete << '\n';
    std::cout << "Unique valid quandles found: " << completeStates.size() << '\n';
    std::cout << "========================================\n";

    for (size_t i = 0; i < completeStates.size(); ++i)
    {
        const State& q = completeStates[i];
        std::cout << "\nQuandle " << (i + 1) << ":\n";
        for (int row = 0; row < q.n; ++row)
        {
            for (int col = 0; col < q.n; ++col)
            {
                std::cout << q.table[stateIndex(row, col, q.n)] << " ";
            }
            std::cout << '\n';
        }
    }
}

int main()
{
    std::cout << "Running verbose CPU-only backtracking search for quandles (n=3)\n\n";
    searchCPU(3);
    return 0;
}
