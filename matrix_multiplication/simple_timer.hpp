#ifndef SIMPLE_TIMER_H
#define SIMPLE_TIMER_H

#include <iostream>
#include <chrono>
#include <map>
#include <string>
#include <stdexcept>

struct TimerData{
    int calls{0};
    size_t time{0};         // Cumulative time in microseconds
    std::string unit{"us"}; // Unit used when printing: "us" or "ms"
};

inline std::map<std::string, TimerData> timing_table;

class SimpleTimer{
public:
    using time_units = std::chrono::microseconds;

    // unit: "us" (microseconds, default) or "ms" (milliseconds), used only for printing
    SimpleTimer(const std::string& name0, const std::string& unit = "us") : name(name0){
        if (unit != "us" && unit != "ms") {
            throw std::invalid_argument("SimpleTimer: unit must be \"us\" or \"ms\", got \"" + unit + "\"");
        }

        if (timing_table.find(name) == timing_table.end()) {
            timing_table[name] = TimerData();
            timing_table[name].unit = unit;
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

   // average time per call for a label, in microseconds
   static double average_us(const std::string& name);

private:
    const std::string name;
    std::chrono::time_point<std::chrono::steady_clock> start_time;

};

inline double SimpleTimer::average_us(const std::string& name){
    const TimerData& data = timing_table.at(name);
    return data.calls > 0 ? static_cast<double>(data.time) / data.calls : 0.0;
}

inline void SimpleTimer::print_timing_results(){
    std::cout << "Timing results:\n";
    for (const auto &entry : timing_table) {
        const TimerData& data = entry.second;
        // convert from microseconds to the requested unit
        double scale = (data.unit == "ms") ? 1e-3 : 1.0;
        std::string unit_label = (data.unit == "ms") ? " ms" : " μs";
        std::cout << entry.first
                  << " -> Total time: " << data.time * scale << unit_label
                  << ", Average time: " << average_us(entry.first) * scale << unit_label
                  << ", Calls: " << data.calls << std::endl;
    }
}

#endif
