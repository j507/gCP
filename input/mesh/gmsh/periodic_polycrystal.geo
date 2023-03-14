// First test of gmsh
Point(1) = {-0.000000, -0.000000, 1.000000} ;
Point(2) = {-0.000000, 0.082377, 1.000000} ;
Point(3) = {0.035345, 0.070369, 1.000000} ;
Point(4) = {0.040589, -0.000000, 1.000000} ;
Point(5) = {0.392699, -0.000000, 1.000000} ;
Point(6) = {0.122013, 0.158649, 1.000000} ;
Point(7) = {0.075122, 0.372613, 1.000000} ;
Point(8) = {-0.000000, 0.402593, 1.000000} ;
Point(9) = {-0.000000, 0.638240, 1.000000} ;
Point(10) = {-0.000000, 0.825768, 1.000000} ;
Point(11) = {-0.000000, 1.000000, 1.000000} ;
Point(12) = {0.325112, 0.146202, 1.000000} ;
Point(13) = {0.398441, 0.031358, 1.000000} ;
Point(14) = {0.698611, 0.100480, 1.000000} ;
Point(15) = {0.711234, -0.000000, 1.000000} ;
Point(16) = {1.000000, -0.000000, 1.000000} ;
Point(17) = {1.000000, 0.082377, 1.000000} ;
Point(18) = {0.246336, 0.479847, 1.000000} ;
Point(19) = {0.400683, 0.324631, 1.000000} ;
Point(20) = {0.258572, 0.568100, 1.000000} ;
Point(21) = {0.062819, 0.737945, 1.000000} ;
Point(22) = {0.044783, 0.826809, 1.000000} ;
Point(23) = {0.052515, 0.839950, 1.000000} ;
Point(24) = {0.040589, 1.000000, 1.000000} ;
Point(25) = {0.540515, 0.362125, 1.000000} ;
Point(26) = {0.726311, 0.148329, 1.000000} ;
Point(27) = {0.763599, 0.162685, 1.000000} ;
Point(28) = {0.908252, 0.439210, 1.000000} ;
Point(29) = {1.000000, 0.402593, 1.000000} ;
Point(30) = {0.599962, 0.507037, 1.000000} ;
Point(31) = {0.425256, 0.675671, 1.000000} ;
Point(32) = {0.445239, 0.787005, 1.000000} ;
Point(33) = {0.374429, 0.900216, 1.000000} ;
Point(34) = {0.392699, 1.000000, 1.000000} ;
Point(35) = {0.735287, 0.555747, 1.000000} ;
Point(36) = {0.893977, 0.469960, 1.000000} ;
Point(37) = {1.000000, 0.638240, 1.000000} ;
Point(38) = {0.731769, 0.836557, 1.000000} ;
Point(39) = {0.749948, 0.819955, 1.000000} ;
Point(40) = {0.711234, 1.000000, 1.000000} ;
Point(41) = {1.000000, 0.825768, 1.000000} ;
Point(42) = {1.000000, 1.000000, 1.000000} ;
Line(1) = {8, 7} ;
Line(2) = {7, 18} ;
Line(3) = {18, 20} ;
Line(4) = {20, 21} ;
Line(5) = {21, 9} ;
Line(6) = {9, 8} ;
Line(7) = {25, 26} ;
Line(8) = {26, 27} ;
Line(9) = {27, 28} ;
Line(10) = {28, 36} ;
Line(11) = {36, 35} ;
Line(12) = {35, 30} ;
Line(13) = {30, 25} ;
Line(14) = {36, 37} ;
Line(15) = {37, 41} ;
Line(16) = {41, 39} ;
Line(17) = {39, 35} ;
Line(18) = {23, 33} ;
Line(19) = {33, 34} ;
Line(20) = {34, 24} ;
Line(21) = {24, 23} ;
Line(22) = {6, 12} ;
Line(23) = {12, 19} ;
Line(24) = {19, 18} ;
Line(25) = {7, 6} ;
Line(26) = {38, 39} ;
Line(27) = {41, 42} ;
Line(28) = {42, 40} ;
Line(29) = {40, 38} ;
Line(30) = {20, 31} ;
Line(31) = {31, 32} ;
Line(32) = {32, 33} ;
Line(33) = {23, 22} ;
Line(34) = {22, 21} ;
Line(35) = {32, 38} ;
Line(36) = {40, 34} ;
Line(37) = {2, 3} ;
Line(38) = {3, 6} ;
Line(39) = {8, 2} ;
Line(40) = {12, 13} ;
Line(41) = {13, 14} ;
Line(42) = {14, 26} ;
Line(43) = {25, 19} ;
Line(44) = {31, 30} ;
Line(45) = {1, 4} ;
Line(46) = {4, 3} ;
Line(47) = {2, 1} ;
Line(48) = {22, 10} ;
Line(49) = {10, 9} ;
Line(50) = {24, 11} ;
Line(51) = {11, 10} ;
Line(52) = {4, 5} ;
Line(53) = {5, 13} ;
Line(54) = {15, 16} ;
Line(55) = {16, 17} ;
Line(56) = {17, 27} ;
Line(57) = {14, 15} ;
Line(58) = {5, 15} ;
Line(59) = {28, 29} ;
Line(60) = {29, 37} ;
Line(61) = {17, 29} ;
Line Loop(62) = {1, 2, 3, 4, 5, 6} ;
Line Loop(63) = {7, 8, 9, 10, 11, 12, 13} ;
Line Loop(64) = {-11, 14, 15, 16, 17} ;
Line Loop(65) = {18, 19, 20, 21} ;
Line Loop(66) = {22, 23, 24, -2, 25} ;
Line Loop(67) = {26, -16, 27, 28, 29} ;
Line Loop(68) = {-4, 30, 31, 32, -18, 33, 34} ;
Line Loop(69) = {-32, 35, -29, 36, -19} ;
Line Loop(70) = {37, 38, -25, -1, 39} ;
Line Loop(71) = {40, 41, 42, -7, 43, -23} ;
Line Loop(72) = {44, -12, -17, -26, -35, -31} ;
Line Loop(73) = {-24, -43, -13, -44, -30, -3} ;
Line Loop(74) = {45, 46, -37, 47} ;
Line Loop(75) = {-5, -34, 48, 49} ;
Line Loop(76) = {-48, -33, -21, 50, 51} ;
Line Loop(77) = {52, 53, -40, -22, -38, -46} ;
Line Loop(78) = {54, 55, 56, -8, -42, 57} ;
Line Loop(79) = {-53, 58, -57, -41} ;
Line Loop(80) = {-10, 59, 60, -14} ;
Line Loop(81) = {-56, 61, -59, -9} ;
 Plane Surface(82) = {62} ;
 Plane Surface(83) = {63} ;
 Plane Surface(84) = {64} ;
 Plane Surface(85) = {65} ;
 Plane Surface(86) = {66} ;
 Plane Surface(87) = {67} ;
 Plane Surface(88) = {68} ;
 Plane Surface(89) = {69} ;
 Plane Surface(90) = {70} ;
 Plane Surface(91) = {71} ;
 Plane Surface(92) = {72} ;
 Plane Surface(93) = {73} ;
 Plane Surface(94) = {74} ;
 Plane Surface(95) = {75} ;
 Plane Surface(96) = {76} ;
 Plane Surface(97) = {77} ;
 Plane Surface(98) = {78} ;
 Plane Surface(99) = {79} ;
 Plane Surface(100) = {80} ;
 Plane Surface(101) = {81} ;
 Physical Surface(0) = {82, 100} ;
 Physical Surface(1) = {83} ;
 Physical Surface(2) = {85, 97} ;
 Physical Surface(3) = {86} ;
 Physical Surface(4) = {88} ;
 Physical Surface(5) = {89, 99} ;
 Physical Surface(6) = {90, 101} ;
 Physical Surface(7) = {91} ;
 Physical Surface(8) = {92} ;
 Physical Surface(9) = {93} ;
 Physical Surface(10) = {84, 95} ;
 Physical Surface(11) = {87, 94, 96, 98} ;
Physical Line(0) = {6, 39, 47, 45, 52, 58, 54, 55, 61, 60, 15, 27, 28, 36, 20, 50, 51, 49} ;

Periodic Line {55} = {47} Translate {1, 0, 0};
Periodic Line {61} = {39} Translate {1, 0, 0};
Periodic Line {60} = {6} Translate {1, 0, 0};
Periodic Line {15} = {49} Translate {1, 0, 0};
Periodic Line {27} = {51} Translate {1, 0, 0};

Periodic Line {50} = {45} Translate {0, 1, 0};
Periodic Line {20} = {52} Translate {0, 1, 0};
Periodic Line {36} = {58} Translate {0, 1, 0};
Periodic Line {28} = {54} Translate {0, 1, 0};

Mesh.Algorithm = 8;
Mesh.RecombinationAlgorithm = 2;
Mesh.RecombineAll = 1;
Mesh.CharacteristicLengthFactor = 0.15;
//Mesh.SubdivisionAlgorithm = 1;
Mesh.Smoothing = 200;
//Mesh.AnisoMax = 1;



Show "*";Translate {0.02, 0.02, 0} {
  Point{23};
}
Translate {0.02, 0.02, 0} {
  Point{23};
}
Translate {0.02, 0.02, 0} {
  Point{31};
}
Translate {0.02, 0.02, 0} {
  Point{31};
}
Translate {0.02, 0.02, 0} {
  Point{32};
}
Translate {-0.02, 0.002, 0} {
  Point{38};
}
Translate {-0.02, 0.002, 0} {
  Point{38};
}
Translate {-0.02, 0.002, 0} {
  Point{38};
}
Translate {-0.02, 0.002, 0} {
  Point{38};
}
Translate {-0.02, -0.002, 0} {
  Point{38};
}
Translate {-0.02, -0.002, 0} {
  Point{38};
}
Translate {-0.02, -0.02, 0} {
  Point{39};
}
Translate {-0.02, -0.02, 0} {
  Point{39};
}
Translate {-0.02, -0.02, 0} {
  Point{39};
}
Translate {-0.02, 0.02, 0} {
  Point{35, 30};
}
Translate {-0.02, 0.02, 0} {
  Point{30};
}
Translate {-0.02, 0.02, 0} {
  Point{35};
}
Translate {-0.02, 0.02, 0} {
  Point{35};
}
Translate {-0.02, 0.02, 0} {
  Point{26};
}
Translate {-0.02, 0.02, 0} {
  Point{14};
}
Translate {0.02, 0, 0} {
  Point{30};
}
Translate {0, 0.02, 0} {
  Point{30};
}
Translate {0, 0.02, 0} {
  Point{30};
}
Translate {0, 0.02, 0} {
  Point{30};
}
Translate {0, 0.02, 0} {
  Point{30};
}
Translate {-0.02, 0.02, 0} {
  Point{30};
}
Translate {-0.02, 0.02, 0} {
  Point{35};
}
Translate {0, -0.02, 0} {
  Point{30};
}
Translate {0, -0.02, 0} {
  Point{28};
}
Translate {0, -0.02, 0} {
  Point{28};
}
Translate {0.02, 0, 0} {
  Point{27};
}
Translate {0.02, 0, 0} {
  Point{27};
}
Translate {0.02, 0, 0} {
  Point{3};
}
