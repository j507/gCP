//+
Geometry.OldNewReg = 0;

radius        = 1.0;
mesh_size     = 1.0;
deg_to_rad    = Pi / 180;
theta         = 60 * deg_to_rad;

n_pattern_columns = 2;
n_pattern_rows    = 2;

n_points_y = 2 * n_pattern_rows + 1;
n_points_x = 4 * n_pattern_columns + 1;

n_horizontal_lines_per_row  = n_pattern_columns * 4;
n_vertical_lines_per_row    = n_pattern_columns * 4 + 1;

n_lines_rows        = 4 * n_pattern_rows + 1;
n_surfaces_rows     = 2 * n_pattern_rows;
n_surfaces_per_row  = 4 * n_pattern_columns;

pattern_x_1[] = {radius, radius*Cos(theta), radius*Cos(theta), radius};
pattern_x_2[] = {radius*Cos(theta), radius, radius, radius*Cos(theta)};

x_coordinate  = 0;
y_coordinate  = 0;
index         = 0;

// Points
For j In {1:n_points_y}
  x_coordinate = 0.0;
  y_coordinate = (j-1) * radius * Sin(theta);
  For i In {1:n_points_x}
    point~{j}~{i} = newp;
    If (i == 1)
      Point(point~{j}~{i}) = {0.0,y_coordinate,0,mesh_size};
    Else
      index = (i < 6) ? (i - 2 ) : (((i - 2) % 4 == 0) ? 0 : index + 1);
      x_coordinate += (j % 2 == 0) ? pattern_x_2[index] : pattern_x_1[index];
      Point(point~{j}~{i}) = {x_coordinate,y_coordinate,0,mesh_size};
    EndIf
  EndFor
EndFor

// Lines
For r In {1:n_lines_rows}
  If (r % 2 == 0)
    first_point = n_points_x * ((r/2)-1) + 1;
    For l In {1:n_vertical_lines_per_row}
      Line(newl) = {first_point, first_point + n_points_x};
      first_point += 1;
    EndFor
  Else
    first_point = n_points_x * (((r+1)/2)-1) + 1;
    For l In {1:n_horizontal_lines_per_row}
      Line(newl) = {first_point, first_point + 1};
      first_point += 1;
    EndFor
  EndIf
EndFor

// Surfaces
For r In {1:n_surfaces_rows}
  first_line = (n_horizontal_lines_per_row + n_vertical_lines_per_row) * (r-1) + 1;
  For s In {1:n_surfaces_per_row}
    line_loop = newll;
    Line Loop(line_loop) =
      {first_line,
        first_line + n_vertical_lines_per_row,
        -(n_vertical_lines_per_row + n_horizontal_lines_per_row + first_line),
        - (first_line + n_horizontal_lines_per_row)};
    Plane Surface(news) = {line_loop};
    first_line += 1;
  EndFor
EndFor


For r In {1:n_surfaces_rows}
  first_physical_group = ((r % 2 == 0) ? r : r-1) * n_pattern_columns + 1;
  second_physical_group = first_physical_group + ((r % 2 == 0) ?  - n_pattern_columns : n_pattern_columns);
  first_index = first_physical_group;
  second_index = second_physical_group;
  //Printf("%g, %g,%g", r,first_physical_group, second_physical_group);
  If (r == 2)
    second_index = 0;
  ElseIf (r == n_surfaces_rows)
    first_index = 1;
  EndIf
  index = 1;
  For s In {1:n_surfaces_per_row}
    If (r == 1)
      If (s == 1)
        Physical Surface(1) = {1};
      ElseIf (s == n_surfaces_per_row)
        Physical Surface(1) += {n_surfaces_per_row};
      Else
        If (s % 2 == 0)
          index += 1;
          Physical Surface(index) = {s};
        Else
          Physical Surface(index) += {s};
        EndIf
      EndIf
    ElseIf (r == 2)
      If (s == 1)
        Physical Surface(first_physical_group) = {n_surfaces_per_row * (r-1) + 1};
      ElseIf (s == n_surfaces_per_row)
        Physical Surface(first_physical_group) += {n_surfaces_per_row * (r-1) + s};
      Else
        If (s % 4 == 0)
          first_index += 1;
          index = first_index;
          Physical Surface(index) = {n_surfaces_per_row * (r-1) + s};
        ElseIf ((s-2) % 4 == 0)
          second_index += 2;
          index = second_index;
          Physical Surface(index) += {n_surfaces_per_row * (r-1) + s};
        Else
          Physical Surface(index) += {n_surfaces_per_row * (r-1) + s};
        EndIf
      EndIf
    ElseIf (r == n_surfaces_rows)
      If (s == 1)
        Physical Surface(1) += {n_surfaces_per_row * (r-1) + s};
      ElseIf (s == n_surfaces_per_row)
        Physical Surface(1) += {n_surfaces_per_row * (r-1) + s};
      Else
        If ( s == 2)
          index = second_physical_group;
          If (r % 2 == 0)
            Physical Surface(index) += {n_surfaces_per_row * (r-1) + s};
          Else
            Physical Surface(index) = {n_surfaces_per_row * (r-1) + s};
          EndIf
        ElseIf (s % 4 == 0)
          first_index += 2;
          index = first_index;
          Physical Surface(index) += {n_surfaces_per_row * (r-1) + s};
        ElseIf ((s-2) % 4 == 0)
          second_index += 1;
          index = second_index;
          If (r % 2 == 0)
            Physical Surface(index) += {n_surfaces_per_row * (r-1) + s};
          Else
            Physical Surface(index) = {n_surfaces_per_row * (r-1) + s};
          EndIf
        Else
          Physical Surface(index) += {n_surfaces_per_row * (r-1) + s};
        EndIf
      EndIf
    Else
      If (s == 1)
        If (r % 2 == 0)
          Physical Surface(first_physical_group) = {(r-1)*n_surfaces_per_row + s};
        Else
          Physical Surface(first_physical_group) += {(r-1)*n_surfaces_per_row + s};
        EndIf
      ElseIf (s == n_surfaces_per_row)
        Physical Surface(first_physical_group) += {(r-1)*n_surfaces_per_row + s};
      Else
        If ( s == 2)
          index = second_physical_group;
          If (r % 2 == 0)
            Physical Surface(index) += {n_surfaces_per_row * (r-1) + s};
          Else
            Physical Surface(index) = {n_surfaces_per_row * (r-1) + s};
          EndIf
        ElseIf (s % 4 == 0)
          first_index += 1;
          index = first_index;
          If (r % 2 == 0)
            Physical Surface(index) = {n_surfaces_per_row * (r-1) + s};
          Else
            Physical Surface(index) += {n_surfaces_per_row * (r-1) + s};
          EndIf
        ElseIf ((s-2) % 4 == 0)
          second_index += 1;
          index = second_index;
          If (r % 2 == 0)
            Physical Surface(index) += {n_surfaces_per_row * (r-1) + s};
          Else
            Physical Surface(index) = {n_surfaces_per_row * (r-1) + s};
          EndIf
        Else
          Physical Surface(index) += {n_surfaces_per_row * (r-1) + s};
        EndIf
        Printf("%g,%g", s, index);
      EndIf
    EndIf
  EndFor
EndFor

/*
//+
Physical Surface(1) = {1};
//+
Physical Surface(2) = {2, 3, 11, 10};
//+
Physical Surface(3) = {4, 5};
//+
Physical Surface(4) = {6, 7, 15, 14};

Physical Surface(1) += {8};
//+
Physical Surface(5) = {9, 17};
//+
Physical Surface(6) = {12, 13, 21, 20};

Physical Surface(5) += {16, 24};
//+
Physical Surface(7) = {18, 19, 27, 26};
//+
Physical Surface(8) = {22, 23, 31, 30};

//+
Physical Surface(9) = {25, 33};
//+
Physical Surface(10) = {28, 29, 37, 36};
//+
Physical Surface(11) = {34, 35, 43, 42};
//+
Physical Surface(12) = {38, 39, 47, 46};
//+
Physical Surface(9) += {32, 40};
//+
Physical Surface(1) += {41, 48};
*/
