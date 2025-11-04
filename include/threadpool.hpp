#pragma once

#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <future>
#include <atomic>

// Simple reusable ThreadPool
// Keeps threads alive across frames, avoiding overhead of creating/destroying threads each frame
class ThreadPool {
public:
    // Constructor: launch 'numThreads' worker threads
    explicit ThreadPool(size_t numThreads);

    // Add a task to the queue
    template<class F>
    void enqueue(F&& f);

    // Wait for all tasks in the queue to complete
    void waitUntilEmpty();

    // Destructor: stop all threads and join
    ~ThreadPool();

private:
    std::vector<std::thread> workers;        // Worker threads
    std::queue<std::function<void()>> tasks; // Task queue
    std::mutex queueMutex;                   // Protect task queue
    std::condition_variable condition;       // Notify workers
    bool stop;                               // Signal to stop workers
	std::condition_variable emptyCondition;  // Notify when queue is empty
	int activeTasks = 0;                     // Count of active tasks
};

inline ThreadPool::ThreadPool(size_t numThreads) : stop(false) {
    for (size_t i = 0; i < numThreads; ++i) {
		// Create worked threads in-place in the workers vector
		// Each thread runs an infinite loop, waiting for tasks to execute
        workers.emplace_back([this]() {
            for (;;) {
                std::function<void()> task;

                // Wait for a task or shutdown signal
                {
					// Lock the queue to avoid race conditions
                    std::unique_lock<std::mutex> lock(queueMutex);
					// Wait until there is a task or we are stopping
                    condition.wait(lock, [this]() { return stop || !tasks.empty(); });

                    // If a stop signal is given and the task queue is empty, ...
                    if (stop && tasks.empty())
                        return; // Exit thread

                    // Move the latest task from the task queue directly in the task variable and remove the queue front
                    task = std::move(tasks.front());
                    tasks.pop();
                }

                // Execute the task
                task();
            }
            });
    }
}

inline ThreadPool& getThreadPool() {
    unsigned int numThreads = std::thread::hardware_concurrency();
    if (numThreads == 0) numThreads = 4; // default to 4 threads if hardware_concurrency cannot determine

	// Create a static ThreadPool instance with the determined number of threads
	// Whenever getThreadPool() is called, the same instance is returned
    static ThreadPool pool(std::thread::hardware_concurrency());
    return pool;
}

inline void ThreadPool::waitUntilEmpty() {
	// Wait until both the task queue is empty and there are no active tasks
    std::unique_lock<std::mutex> lock(queueMutex);
    emptyCondition.wait(lock, [this]() { return tasks.empty() && (activeTasks == 0); });
}

// F&& f is a universal reference (can bind to lvalues and rvalues)
template<class F>
inline void ThreadPool::enqueue(F&& f) {
    {
        std::unique_lock<std::mutex> lock(queueMutex);
		// Add a new task to the queue
		// By using a lambda, we can wrap the original task 'f' to manage activeTasks count
		// std::forward<F>(f) preserves the value category (lvalue/rvalue) of 'f'
        tasks.emplace([this, f = std::forward<F>(f)]() {
            {
                std::unique_lock<std::mutex> lock(queueMutex);
                ++activeTasks;
            }

            f(); // run the real task

            {
                std::unique_lock<std::mutex> lock(queueMutex);
                --activeTasks;
                if (tasks.empty() && activeTasks == 0)
                    emptyCondition.notify_all();
            }
            });
    }

	// Notify one worker thread that a new task is available
    condition.notify_one();
}

inline ThreadPool::~ThreadPool() {
    {
        std::unique_lock<std::mutex> lock(queueMutex);
        stop = true;
    }
	// Notify all threads to wake up and exit
    condition.notify_all();
    for (auto& t : workers)
		t.join(); // Let the threads finish their execution, and join them back to the main thread
}
