PROJECT(tests)

INCLUDE_DIRECTORIES(
    include
    ${CMAKE_SOURCE_DIR}/include
    )

SET(SOURCE_FILES
    constitutive_laws_test.cc
    crystal_data_test.cc
    fe_collection_test.cc
    )

FOREACH(sourcefile ${SOURCE_FILES})
    # string replace: cut off .cc from files
    STRING(REPLACE ".cc" "" executablename ${sourcefile} )
    PROJECT(${executablename})
    ADD_EXECUTABLE(${executablename} ${sourcefile})
    DEAL_II_SETUP_TARGET(${executablename})
    TARGET_LINK_LIBRARIES(${executablename} gCP stdc++fs)
ENDFOREACH(sourcefile ${APP_SOURCES})
