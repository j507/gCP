#PROJECT(tests)
#
#INCLUDE_DIRECTORIES(
#    include
#    ${CMAKE_SOURCE_DIR}/include
#    )
#
#SET(SOURCE_FILES
#    compute_trial_microstress.cc
#    constitutive_laws_test.cc
#    crystal_data_test.cc
#    fe_collection_test.cc
#    line_search_test.cc
#    make_periodicity_constraints.cc
#    quadrature_point_history_test.cc
#    mark_interface_test.cc
#    )
#
#FOREACH(sourcefile ${SOURCE_FILES})
#    # string replace: cut off .cc from files
#    STRING(REPLACE ".cc" "" executablename ${sourcefile} )
#    PROJECT(${executablename})
#    ADD_EXECUTABLE(${executablename} ${sourcefile})
#    DEAL_II_SETUP_TARGET(${executablename})
#    TARGET_LINK_LIBRARIES(${executablename} gCP stdc++fs)
#ENDFOREACH(sourcefile ${APP_SOURCES})

SET(TEST_LIBRARIES gCP)
DEAL_II_PICKUP_TESTS()
