# Additional clean files
cmake_minimum_required(VERSION 3.16)

if("${CONFIG}" STREQUAL "" OR "${CONFIG}" STREQUAL "Debug")
  file(REMOVE_RECURSE
  "CMakeFiles\\simon_autogen.dir\\AutogenUsed.txt"
  "CMakeFiles\\simon_autogen.dir\\ParseCache.txt"
  "simon_autogen"
  )
endif()
