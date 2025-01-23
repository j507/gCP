SET(SOURCE_FILES
    simple_shear.cc
    jacobian.cc
    )

FOREACH(sourcefile ${SOURCE_FILES})
  STRING(REPLACE ".cc" "" executablename ${sourcefile} )
  PROJECT(${executablename})
  ADD_EXECUTABLE(${executablename} ${sourcefile})
  DEAL_II_SETUP_TARGET(${executablename})
  TARGET_LINK_LIBRARIES(${executablename} gCP stdc++fs)
ENDFOREACH(sourcefile ${APP_SOURCES})