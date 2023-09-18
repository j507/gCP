//+
Geometry.OldNewReg = 0;

radius        = 1.0;
mesh_size     = 1.0;
deg_to_rad    = Pi / 180;
theta         = 60 * deg_to_rad;

n_pattern_columns = 2;
n_pattern_rows    = 6;

n_points_y = 2 * n_pattern_rows + 1;
n_points_x = 4 * (n_pattern_columns - 1) + 7;

n_horizontal_lines_per_row  = 4 * (n_pattern_columns - 1) + 6;
n_vertical_lines_per_row    = 4 * (n_pattern_columns - 1) + 7;

n_lines_rows        = 4 * n_pattern_rows + 1;
n_surfaces_rows     = 2 * n_pattern_rows;
n_surfaces_per_row  = 4 * (n_pattern_columns - 1) + 6;

x_increments_0[] = {radius, radius*Cos(theta), radius*Cos(theta), radius, radius, radius*Cos(theta)};
y_increments_0[] = {radius*Cos(theta), radius, radius, radius*Cos(theta), radius*Cos(theta), radius};

x_increments_1[] = {radius*Cos(theta), radius, radius, radius*Cos(theta)};
y_increments_1[] = {radius, radius*Cos(theta), radius*Cos(theta), radius};

x_coordinate  = 0;
y_coordinate  = 0;
index         = 0;

// Points
For j In {1:n_points_y}
  x_coordinate = 0.0;
  y_coordinate = (j-1) * radius * Sin(theta);
  For i In {1:n_points_x}
    If (i == 1)
      Point(newp) = {0.0,y_coordinate,0,mesh_size};
    ElseIf (i <= 7)
      index = i-2;
      x_coordinate += (j % 2 == 0) ? y_increments_0[index] : x_increments_0[index];
      Point(newp) = {x_coordinate,y_coordinate,0,mesh_size};
    Else
      If ( i % 4 == 0)
        index = 0;
      Else
        index += 1;
      EndIf
      x_coordinate += (j % 2 == 0) ? y_increments_1[index] : x_increments_1[index];
      Point(newp) = {x_coordinate,y_coordinate,0,mesh_size};
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
  first_physical_group = ((r % 2 == 0) ? r : r-1) * (n_pattern_columns + 1) + 1;
  second_physical_group = first_physical_group + ((r % 2 == 0) ?  - (n_pattern_columns + 1) : (n_pattern_columns + 1));
  first_index = first_physical_group;
  second_index = second_physical_group;
  index = 1;
  If (r == 2)
    second_index = 0;
  EndIf
  For s In {1:n_surfaces_per_row}
    If (r == 1)
      If (s == 1)
        Physical Surface(1) = {1};
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
    Else
      If (s == 1)
        If (r % 2 == 0)
          Physical Surface(first_physical_group) = {(r-1)*n_surfaces_per_row + s};
        Else
          Physical Surface(first_physical_group) += {(r-1)*n_surfaces_per_row + s};
        EndIf
      Else
        If (s == 2)
          index = second_physical_group;
          If (r % 2 == 0)
            Physical Surface(index) += {(r-1)*n_surfaces_per_row + s};
          Else
            Physical Surface(index) = {(r-1)*n_surfaces_per_row + s};
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
      EndIf
    EndIf
  EndFor
EndFor//+

//Transfinite Curve {53} = 10 Using Progression 1;

//Mesh.Algorithm = 4;
//Mesh.RecombinationAlgorithm = 2;
//Mesh.RecombineAll = 1;
Mesh.CharacteristicLengthFactor = 0.30;
//Mesh.SubdivisionAlgorithm = 1;
Mesh.Smoothing = 200;
//Mesh.AnisoMax = 1;