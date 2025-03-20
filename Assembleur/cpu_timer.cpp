/*
 * cpu_timer.cpp
 *
 *   Created on: Apr 2, 2015
 *  Modified on: Feb, 2022
 *       Author: Jean-Michel Richer
 */

#include "cpu_timer.h"


CPUTimer::Clocks CPUTimer::get_clocks() {

	return std::chrono::steady_clock::now();
	
}

CPUTimer::CPUTimer() {

	_event_stop = _event_start = get_clocks();
	
}


void CPUTimer::start() {

	_event_start = get_clocks();
	
}


void CPUTimer::stop() {

	_event_stop = get_clocks();

}


ostream& CPUTimer::print(ostream& out) {

	auto millis = _event_stop - _event_start;
	
	std::chrono::hours   hh = std::chrono::duration_cast<std::chrono::hours>(millis);
	std::chrono::minutes mm = std::chrono::duration_cast<std::chrono::minutes>(millis % chrono::hours(1));
	std::chrono::seconds ss = std::chrono::duration_cast<std::chrono::seconds>(millis % chrono::minutes(1));
	milli_sec msec = std::chrono::duration_cast<milli_sec>(millis % chrono::seconds(1));
	out << setfill('0') << setw(2) << hh.count() << "::"
			<< setw(2) << mm.count() << "::"
			<< setw(2) << ss.count() << "::"
			<< setw(3) << msec.count();

	double total_in_ms = (ss.count() + 60 * mm.count() + 3600 * hh.count()) * 1000 + msec.count();
	
	out << "|" << total_in_ms;
	double total_in_s = (ss.count() + 60 * mm.count() + 3600 * hh.count()) + (msec.count() / 1000.0);
	out << "|" << total_in_s;
	
	return out;
	
}


long_integer CPUTimer::ns() {

	auto diff = _event_stop - _event_start;
	nano_sec nsec = std::chrono::duration_cast<nano_sec>( diff );
	
	return static_cast<long int>( nsec.count() );
	
}


long_integer CPUTimer::us() {

	auto diff = _event_stop - _event_start;
	micro_sec usec = std::chrono::duration_cast<micro_sec>( diff );
	
	return static_cast<long int>( usec.count() );
	
}


long_integer CPUTimer::ms() {

	auto diff = _event_stop - _event_start;
	milli_sec msec = std::chrono::duration_cast<milli_sec>( diff );
	
	return static_cast<long int>( msec.count() );

}


long_integer CPUTimer::s() {

	auto diff = _event_stop - _event_start;
	std::chrono::seconds sec = std::chrono::duration_cast<std::chrono::seconds>( diff );
	
	return static_cast<long int>( sec.count() );

}


long_integer CPUTimer::s_elapsed() {

	auto diff = std::chrono::steady_clock::now() - _event_start;
	std::chrono::seconds sec = std::chrono::duration_cast<std::chrono::seconds>( diff );
	
	return static_cast<long int>( sec.count() );

}

