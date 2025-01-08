# Additional clean files
cmake_minimum_required(VERSION 3.16)

if("${CONFIG}" STREQUAL "" OR "${CONFIG}" STREQUAL "Debug")
  file(REMOVE_RECURSE
  "CMakeFiles\\tp8_autogen.dir\\AutogenUsed.txt"
  "CMakeFiles\\tp8_autogen.dir\\ParseCache.txt"
  "tp8_autogen"
  )
endif()
