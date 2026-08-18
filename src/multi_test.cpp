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

    // 5. Self-distributivity: (a*b)*c = (a*c)*(b*c)
    for (int a = 0; a < s.n; ++a)
    {
        for (int b = 0; b < s.n; ++b)
        {
            if (!isAssigned(s, a, b))
                continue;

            for (int c = 0; c < s.n; ++c)
            {
                if (!isAssigned(s, a, c))
                    continue;
                if (!isAssigned(s, b, c))
                    continue;

                int ab = getCell(s, a, b);
                if (ab < 0 || ab >= s.n)
                    return false;
                if (!isAssigned(s, ab, c))
                    continue;
                
                int ab_c = getCell(s, ab, c);
                
                int ac = getCell(s, a, c);
                int bc = getCell(s, b, c);
                if (ac < 0 || ac >= s.n || bc < 0 || bc >= s.n)
                    return false;
                if (!isAssigned(s, ac, bc))
                    continue;
                
                int ac_bc = getCell(s, ac, bc);
                
                if (ab_c != ac_bc)
                    return false;
            }
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

            for (int c = 0; c < s.n; ++c)
            {
                const int ab = getCell(s, a, b);
                const int ab_c = getCell(s, ab, c);
                const int ac = getCell(s, a, c);
                const int bc = getCell(s, b, c);
                const int ac_bc = getCell(s, ac, bc);
                if (ab_c != ac_bc)
                    return false;
            }
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

void searchCPU(int n)
{
    std::vector<State> stack = buildInitialRoots(n);
    std::unordered_set<std::string> seen;
    std::vector<State> completeStates;
    int nodesExplored = 0;

    while (!stack.empty())
    {
        State s = stack.back();
        stack.pop_back();
        nodesExplored++;

        if (!isFeasible(s))
            continue;

        if (isComplete(s))
        {
            if (isValidQuandle(s))
            {
                const std::string key = canonicalize(s);
                if (seen.insert(key).second)
                {
                    completeStates.push_back(s);
                }
            }
            continue;
        }

        const Cell cell = chooseNextUnassignedCell(s);
        if (cell.row == -1)
            continue;

        const std::vector<int> candidates = generateCandidates(s, cell);
        for (int value : candidates)
        {
            State child = applyAssignment(s, cell, value);
            if (isFeasible(child))
                stack.push_back(child);
        }
    }

    std::cout << "Order " << n << ": " << completeStates.size() << " quandles found (explored " << nodesExplored << " nodes)\n";
}

int main()
{
    std::cout << "Testing quandle enumeration algorithm:\n";
    searchCPU(1);
    searchCPU(2);
    searchCPU(3);
    searchCPU(4);
    return 0;
}
