#ifndef SIMPLE_TIMER_H
#define SIMPLE_TIMER_H

#include <iostream>
#include <chrono>
#include <map>
#include <string>

struct TimerData{
    int calls{0};
    size_t time{0}; // Time in milliseconds
};

std::map<std::string, TimerData> timing_table;

class SimpleTimer{
public:
    using time_units = std::chrono::microseconds;
    
    SimpleTimer(const std::string& name0) : name(name0){
        if (timing_table.find(name) == timing_table.end()) {
            timing_table[name] = TimerData();
        }
        
        // Increment the call count for this timing label
        timing_table[name].calls++;

        start_time = std::chrono::steady_clock::now();
    }

    ~SimpleTimer(){
        auto end_time = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<time_units>(end_time - start_time).count();
        
        // Update the cumulative time for this label
        timing_table[name].time += duration; 
    }

   //here comes important new info: static class functions - they can be called without the object of the class
   static void print_timing_results();

private:
    const std::string name;
    std::chrono::time_point<std::chrono::steady_clock> start_time;

};


void SimpleTimer::print_timing_results(){
    std::cout << "Timing results:\n";
    for (const auto &entry : timing_table) {
        std::cout << entry.first << " -> Total time: " << entry.second.time << " μs, Calls: " << entry.second.calls << std::endl;
    }
}

#endif