#include <iostream>
#include <string>
#include <unordered_set>
#include <vector>

// Brute force checker: try all n^(n²) possible tables and validate
bool isValidQuandleBruteForce(const int table[9], int n)
{
    // 1. Idempotence: a*a = a for all a
    for (int a = 0; a < n; ++a)
    {
        if (table[a * n + a] != a)
            return false;
    }

    // 2. Right division: (a*b)*b = a for all a, b
    for (int a = 0; a < n; ++a)
    {
        for (int b = 0; b < n; ++b)
        {
            int ab = table[a * n + b];
            int ab_b = table[ab * n + b];
            if (ab_b != a)
                return false;
        }
    }

    // 3. Row uniqueness (each row is a permutation)
    for (int r = 0; r < n; ++r)
    {
        int seen[9] = {0};
        for (int c = 0; c < n; ++c)
        {
            int val = table[r * n + c];
            if (val < 0 || val >= n || seen[val])
                return false;
            seen[val] = 1;
        }
    }

    // 4. Column uniqueness (each column is a permutation)
    for (int c = 0; c < n; ++c)
    {
        int seen[9] = {0};
        for (int r = 0; r < n; ++r)
        {
            int val = table[r * n + c];
            if (val < 0 || val >= n || seen[val])
                return false;
            seen[val] = 1;
        }
    }

    return true;
}

std::string tableToString(const int table[9])
{
    std::string s;
    for (int i = 0; i < 9; ++i)
    {
        s.push_back('0' + table[i]);
    }
    return s;
}

int main()
{
    for (int order = 1; order <= 3; ++order)
    {
        std::unordered_set<std::string> found;
        long long maxTables = 1;
        for (int i = 0; i < order * order; ++i)
            maxTables *= order;

        std::cout << "Testing order " << order << " (" << maxTables << " tables to check)...\n";

        std::vector<int> table(order * order);

        // Generate all tables using base-order enumeration
        for (long long i = 0; i < maxTables; ++i)
        {
            long long temp = i;
            for (int j = 0; j < order * order; ++j)
            {
                table[j] = temp % order;
                temp /= order;
            }

            if (order == 1 && i == 0)
            {
                std::cout << "  Testing order 1 table: [0]\n";
                bool valid = isValidQuandleBruteForce(table.data(), order);
                std::cout << "    Result: " << (valid ? "VALID" : "INVALID") << "\n";
            }

            if (isValidQuandleBruteForce(table.data(), order))
            {
                std::string canonical(table.begin(), table.end());
                found.insert(canonical);
            }
        }

        std::cout << "  Found " << found.size() << " quandles of order " << order << "\n";
        if (found.size() <= 3 && found.size() > 0)
        {
            int count = 0;
            for (const auto& q : found)
            {
                std::cout << "  Quandle #" << (++count) << ":\n";
                for (int r = 0; r < order; ++r)
                {
                    std::cout << "    ";
                    for (int c = 0; c < order; ++c)
                    {
                        std::cout << (char)('0' + q[r * order + c]) << " ";
                    }
                    std::cout << "\n";
                }
            }
        }
        std::cout << "\n";
    }

    return 0;
}
