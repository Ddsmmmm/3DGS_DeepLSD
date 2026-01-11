# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

file(MAKE_DIRECTORY
  "/home/ddsm/DownLoad/DeepLSD/third_party/homography_est/build/temp.linux-x86_64-cpython-38/_deps/googletest-src"
  "/home/ddsm/DownLoad/DeepLSD/third_party/homography_est/build/temp.linux-x86_64-cpython-38/_deps/googletest-build"
  "/home/ddsm/DownLoad/DeepLSD/third_party/homography_est/build/temp.linux-x86_64-cpython-38/_deps/googletest-subbuild/googletest-populate-prefix"
  "/home/ddsm/DownLoad/DeepLSD/third_party/homography_est/build/temp.linux-x86_64-cpython-38/_deps/googletest-subbuild/googletest-populate-prefix/tmp"
  "/home/ddsm/DownLoad/DeepLSD/third_party/homography_est/build/temp.linux-x86_64-cpython-38/_deps/googletest-subbuild/googletest-populate-prefix/src/googletest-populate-stamp"
  "/home/ddsm/DownLoad/DeepLSD/third_party/homography_est/build/temp.linux-x86_64-cpython-38/_deps/googletest-subbuild/googletest-populate-prefix/src"
  "/home/ddsm/DownLoad/DeepLSD/third_party/homography_est/build/temp.linux-x86_64-cpython-38/_deps/googletest-subbuild/googletest-populate-prefix/src/googletest-populate-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/home/ddsm/DownLoad/DeepLSD/third_party/homography_est/build/temp.linux-x86_64-cpython-38/_deps/googletest-subbuild/googletest-populate-prefix/src/googletest-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/home/ddsm/DownLoad/DeepLSD/third_party/homography_est/build/temp.linux-x86_64-cpython-38/_deps/googletest-subbuild/googletest-populate-prefix/src/googletest-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
