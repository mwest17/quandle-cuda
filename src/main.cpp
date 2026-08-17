#include <queue>
#include <pthread.h>
// #include <eigen> // ehhhh only because I need to bitpack most likely and I would like this to run on CUDA with little modifications

#include "quandle.h"

const int N = 3;

std::queue<Quandle<N, unsigned short>*> frontier; // Must be FIFO for BFS!!!

int main()
{
    // For Cohen Quandles
    // std::vector<int> factors;
    // for (int i = 0; i < N; ) // Only quandle with N orbits of order N is the trivial
    // {
    //     if (N % i == 0)
    //     {
    //         factors.push_back(i);
    //     }
    // }
    
    // Init with all orbits that add up to N
        // Exceptions are full connected of order N and trivial
}




// thread safe bfs search pattern

// frontier should have t
// for 1 billion elements on the frontier and 14x14 matrices, we need 144 gb of vram
// bitpack??? 3-4 bits per index. 


// I think DFS tree search
// If I do worked managed local queue, how to load balance though? This would be preferred though.
    // Also how to reduce divergence in the warp
    // What about a frontier just in shared memory at the warp level (or block level). Or just per thread


// TODO**
    // will memory access for queue be ok? Make sure bulk allocate in advance. Only dyanmically allocate if we run out of space
    // improve memory access pattern for CUDA.
    
    // I should identify serialization concerns and divergence concerns

    // What to do if we run out of memory. 
        // Do all threads switch to DFS until we have enough memory for next BFS queue expansion?? This will be worse for preventing divergence
        // Could we instead offload part of the frontier to CPU memory. Eh only pushes the problem back a step. We will run out of cpu memory soon too. Plus that would be very slow
        // 

    // Need to do all pruning possible even if at cost of divergence as it eliminates so many possible states
template <typename T>
void dispatch(std::queue<T> frontier, std::mutex mutex)
{
    while (true)
    {
        mutex.lock() // No lock, just retrieve from frontier based on thread index!!!
        if (!frontier.empty())
        {
            mutex.unlock();
            break; // end of entire program since all of frontier has been explored
        } 

        T& element = frontier.pop(); // Don't want to have to take this memory creation cost again. Ensure this is just a pointer
        mutex.unlock();
 
        if (finished(element))
        {
            // Do isomorphism check during insertion or during a 2nd pass algorithm?
                // On the fly would save on memory usage and memory usage is a serious concern
                // Irrelevant for compute cost
                // But on the fly would also cause more divergence
            // save to finished queue
            continue;
        }

        // if frontier too full based on estimate
            // call dfs routine
            // we still need some amount of memory for dfs for paths not explored

        // what if all threads pop and push to front of the queue. Decreasing time to reach complete generations. 
        // What would divergence issues be?
            // I don't think many memory access issues (maybe issues if we don't copy arrays to shared)
            
            // all threads take a partial matrix (fine)
            // Check if matrix is complete (some divergence) - Can we order it so the matrices that are near complete are taken by the same warp
            // expand state (some divergence), as there will be a different number of child states depending on the matrix
                // - Can we maybe order the ones.
                // I'll need to make sure that partial matrices that are closely 'related' to each other are placed on the froniter nearby each other



        std::array<T> children = expand(element); // Ensure we only try the automorphisms of the orbit elements

        mutex.lock()
        for (T child : children)
        {
            frontier.push_back(child);    
        }
        mutex.unlock();
    }

}


// do branching + expansion on CPU
// do evaluation of quandle axioms (and filled in implied sections?) on GPU
// CPU then recieves list of which ones to prune
// CPU then continues search (DFS) on just one node, but entire frontier is given to GPU


// Maybe look into CUDA unified memory on where to store the FIFO queue

// Perform DFS on CPU until FIFO queue has at least the number of threads present (times some scalar factor to amortize copying cost)
// Run GPU kernel on all nodes in FIFO (memcpy part of fifo queue)
    // fill in any implied positions
    // evaluate if these states can become valid quandles
    // somehow signal CPU which are bad and which are good
// CPU then removes the invalid states and adds new expanded states to fifo
// CPU continues DFS

// literally see cuda code in paper!!!!!!!!!
// and on github, I could literally kiss them


// I should probably consider using unified memory as a nice abstraction for this

// CPU
// queue_filo frontier 
// if len(frontier) > num_threads * amortize_const
//  remove first num_threads * amortize_const from frontier and send to gpu memory
//  retrieve new nodes from gpu
// else
//  pop from frontier
//  ensure node is worthwhile  
//  if node is final
//   add to finished list
//  else  
//   explore node
//   push new nodes to frontier
// loop

void explore()
{
    constexpr int NUM_THREADS = 1024; // Take from CUDA code
    constexpr int AMORTIZE_FACTOR = 10;
    queue_fifo frontier; // TODO**
    std::vector<State> finished;

    initFrontier(frontier); // place all possible combos of quandles as orbits

    while (frontier.size() > 0)
    {
        if (froniter.size() > NUM_THREADS * AMORTIZE_FACTOR)
        {
            constexpr size_t blockSize = sizeof(State) * NUM_THREADS * AMORTIZE_FACTOR;
            cudaMemCopy(frontier.data() + (sizeof(State) * frontier.size()) - (blockSize), blockSize);
            // Call CUDA kernel
            // TODO**
            
            cudaMemCopy();

            // for all states in gpu memory
                // push states into frontier
        }
        else
        {
            State& s = froniter.pop();
            if (isValid(s))
            {
                if (finished(s))
                {
                    finished.push_back(s);
                }
                else
                {
                    std::vector<State> children = explore(s);
                    for (State child : children)
                    {
                        frontier.push(child);
                    }
                }
            }   
            else
            {
                delete s;
            }
        }
    }
}


// GPU
// global_memory nodes
// for i in range(amortize_const)
//  check if node is worthwhile (ie check if it satisfies all axioms, and doesn't have any contradictions)
//  if finished
//   save to finished list
//  else
//   explore node (by generating all permutations of the orbits) (somewhat worried about divergence)
//   add new nodes to list sent to cpu

// how to prevent serialization issues when writing data to send to cpu?
    // allocate enough memory for worst possible case and have each thread only write to its special section?
        // too much memory?
        // I would need to do this anyway, right? Since otherwise I could not have enough memory worst case
        // I would just need a bit in the State struct to indicate if it was valid or not
        // I want to CPU memory to just be an array of states if possible. saves on the headache


void branchKernel()
{
    for (int i = 0; i < AMORTIZE_FACTOR; i++)
    {
        State s = frontier[threadNum * i];
        if (isValid(s))
        {
            if (isFinished(s))
            {
                // ehh not sure
                // at minimum toCpu.push_back(s);
            }
            else
            {
                std::vector<State> children = explore(s);
                for (State child : children)
                {
                    toCpu.push_back(child);
                }
            }
        }
    }
}