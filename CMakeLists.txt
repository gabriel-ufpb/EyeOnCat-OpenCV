cmake_minimum_required(VERSION 3.16)
project(eye_cat_v3 LANGUAGES CXX)

# ------------------------------------------------------------------
# Configurações básicas
# ------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# ------------------------------------------------------------------
# Dependências
# ------------------------------------------------------------------

# OpenCV (usa o módulo "geometry", presente a partir do OpenCV 5)
find_package(OpenCV REQUIRED)

# NCNN
# Se o CMake não achar automaticamente, informe o caminho do ncnnConfig.cmake
# ao gerar o build, ex:
#   cmake -B build -DNCNN_DIR=/caminho/para/ncnn/install/lib/cmake/ncnn
find_package(ncnn REQUIRED)

# Vulkan (necessário para a inferência acelerada por GPU no NCNN)
find_package(Vulkan REQUIRED)

# Threads (std::thread usado para o alerta sonoro em background)
find_package(Threads REQUIRED)

# ------------------------------------------------------------------
# Executável
# ------------------------------------------------------------------
add_executable(eye_cat_v3 mainT.cpp)

target_include_directories(eye_cat_v3 PRIVATE
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(eye_cat_v3 PRIVATE
    ${OpenCV_LIBS}
    ncnn
    Vulkan::Vulkan
    Threads::Threads
)

# ------------------------------------------------------------------
# Mensagens de ajuda ao configurar
# ------------------------------------------------------------------
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "Build type:     ${CMAKE_BUILD_TYPE}")
