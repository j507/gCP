
#include <gCP/finite_domains.h>


#include <deal.II/distributed/tria.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/numerics/data_out.h>

#include <iostream>
#include <fstream>
#include <cmath>



namespace gCP
{



namespace FiniteDomains
{



namespace internal
{



void plate_with_notches(
  dealii::parallel::distributed::Triangulation<2> &triangulation)
{
  dealii::parallel::distributed::Triangulation<2>
    lower_square(MPI_COMM_WORLD), upper_square(MPI_COMM_WORLD),
      left_notch(MPI_COMM_WORLD), right_notch(MPI_COMM_WORLD);

  dealii::GridGenerator::hyper_cube_with_cylindrical_hole(
    left_notch, 0.5, 1);

  dealii::GridGenerator::hyper_cube_with_cylindrical_hole(
    right_notch, 0.5, 1);

  std::vector<unsigned int> n_repetitions(2, 2);

  dealii::GridGenerator::subdivided_hyper_rectangle(
    lower_square,
    n_repetitions,
    dealii::Point<2>(-1.,-1.),
    dealii::Point<2>(+1.,+1.));

  dealii::GridTools::shift(dealii::Point<2>(0.,-2.), lower_square);

  dealii::GridGenerator::subdivided_hyper_rectangle(
    upper_square,
    n_repetitions,
    dealii::Point<2>(-1,-1),
    dealii::Point<2>(+1,+1));

  dealii::GridTools::shift(dealii::Point<2>(0.,2.), upper_square);

  std::set<typename
    dealii::parallel::distributed::Triangulation<2>::
      active_cell_iterator>
        cells_to_remove;

  for (const auto &active_cell : left_notch.active_cell_iterators())
  {
    if (active_cell->center()[0] < 0.)
    {
      cells_to_remove.insert(active_cell);
    }
  }

  dealii::GridGenerator::create_triangulation_with_removed_cells(
  left_notch,
  cells_to_remove,
  left_notch);

  dealii::GridTools::shift(dealii::Point<2>(-1.,0.), left_notch);

  cells_to_remove.clear();

  for (const auto &active_cell : right_notch.active_cell_iterators())
  {
    if (active_cell->center()[0] > 0.)
    {
      cells_to_remove.insert(active_cell);
    }
  }

  dealii::GridGenerator::create_triangulation_with_removed_cells(
    right_notch,
    cells_to_remove,
    right_notch);

  dealii::GridTools::shift(dealii::Point<2>(1.0,0.), right_notch);

  auto min_line_length = [](
    const dealii::parallel::distributed::Triangulation<2> &tria)->double
  {
    double length = std::numeric_limits<double>::max();

    for (const auto &cell : tria.active_cell_iterators())
    {
      for (const auto n : cell->line_indices())
      {
        length = std::min(
          length,
          (cell->line(n)->vertex(0) - cell->line(n)->vertex(1)).norm());
      }
    }
    return length;
  };

  auto compute_tolerance = [min_line_length](
    const dealii::parallel::distributed::Triangulation<2> &tria1,
    const dealii::parallel::distributed::Triangulation<2> &tria2)->double
  {
    return .5 * std::min(min_line_length(tria1), min_line_length(tria2));
  };

  dealii::GridGenerator::merge_triangulations(
    left_notch,
    right_notch,
    triangulation,
    compute_tolerance(left_notch, right_notch));

  dealii::GridGenerator::merge_triangulations(
    triangulation,
    upper_square,
    triangulation,
    compute_tolerance(triangulation, upper_square));

  dealii::GridGenerator::merge_triangulations(
    triangulation,
    lower_square,
    triangulation,
    compute_tolerance(triangulation, lower_square));

  triangulation.reset_all_manifolds();
}



}; // namespace internal



void plate_with_notches(
  dealii::parallel::distributed::Triangulation<2> &triangulation,
  const unsigned int n_slices,
  const double depth)
{
  (void)n_slices;
  (void)depth;

  internal::plate_with_notches(triangulation);

  const double width = 1.0,
               height = 3. * width,
               radius = 0.5 * width;

  const double tolerance = 0.01;

  const unsigned int
    x_lower_boundary_id = 0,
    x_upper_boundary_id = 1,
    y_lower_boundary_id = 2,
    y_upper_boundary_id = 3,
    left_notch_boundary_id = 4,
    right_notch_boundary_id = 5;

  for (const auto &active_cell :
        triangulation.active_cell_iterators())
  {
    for (const auto &face_index : active_cell->face_indices())
    {
      if (active_cell->face(face_index)->at_boundary() &&
            active_cell->is_locally_owned())
      {
        const dealii::Point<2> face_center =
          active_cell->face(face_index)->center();

        if (face_center[0] < -(width - tolerance))
        {
          active_cell->face(face_index)->
            set_boundary_id(x_lower_boundary_id);
        }
        else if (face_center[0] > (width - tolerance))
        {
          active_cell->face(face_index)->
            set_boundary_id(x_upper_boundary_id);
        }
        else if (face_center[1] < -(height - tolerance))
        {
          active_cell->face(face_index)->
            set_boundary_id(y_lower_boundary_id);
        }
        else if (face_center[1] > (height - tolerance))
        {
          active_cell->face(face_index)->
            set_boundary_id(y_upper_boundary_id);
        }

        if (face_center[1] < (radius - tolerance) &&
              face_center[1] > -(radius - tolerance))
        {
          active_cell->face(face_index)->set_boundary_id(
            face_center[0] < 0. ?
              left_notch_boundary_id :
              right_notch_boundary_id);
        }
      }
    }
  }

  dealii::Point<2>
    left_notch_center(-1.,0.),
    right_notch_center(+1.,0.);

  const dealii::SphericalManifold<2>
    left_manifold(left_notch_center),
    right_manifold(right_notch_center);

  triangulation.set_all_manifold_ids_on_boundary(4, 4);

  triangulation.set_all_manifold_ids_on_boundary(5, 5);

  triangulation.set_manifold(4, left_manifold);

  triangulation.set_manifold(5, right_manifold);
}



void plate_with_notches(
  dealii::parallel::distributed::Triangulation<3> &triangulation,
  const unsigned int n_slices,
  const double depth)
{
  const double width = 1.0,
               height = 3. * width,
               radius = 0.5 * width;

  const double tolerance = 0.01;

  dealii::parallel::distributed::Triangulation<2>
    internal_triangulation(MPI_COMM_WORLD);

  internal::plate_with_notches(internal_triangulation);

  dealii::GridGenerator::extrude_triangulation(
    internal_triangulation,
    n_slices,
    depth,
    triangulation);

  const unsigned int
    x_lower_boundary_id = 0,
    x_upper_boundary_id = 1,
    y_lower_boundary_id = 2,
    y_upper_boundary_id = 3,
    z_lower_boundary_id = 4,
    z_upper_boundary_id = 5,
    left_notch_boundary_id = 6,
    right_notch_boundary_id = 7;

  for (const auto &active_cell :
        triangulation.active_cell_iterators())
  {
    for (const auto &face_index : active_cell->face_indices())
    {
      if (active_cell->face(face_index)->at_boundary() &&
            active_cell->is_locally_owned())
      {
        const dealii::Point<3> face_center =
          active_cell->face(face_index)->center();

        if (face_center[0] < -(width - tolerance))
        {
          active_cell->face(face_index)->
            set_boundary_id(x_lower_boundary_id);
        }
        else if (face_center[0] > (width - tolerance))
        {
          active_cell->face(face_index)->
            set_boundary_id(x_upper_boundary_id);
        }
        else if (face_center[1] < -(height - tolerance))
        {
          active_cell->face(face_index)->
            set_boundary_id(y_lower_boundary_id);
        }
        else if (face_center[1] > (height - tolerance))
        {
          active_cell->face(face_index)->
            set_boundary_id(y_upper_boundary_id);
        }
        else if (face_center[2] < (tolerance))
        {
          active_cell->face(face_index)->
            set_boundary_id(z_lower_boundary_id);
        }
        else if (face_center[2] > (depth - tolerance))
        {
          active_cell->face(face_index)->
            set_boundary_id(z_upper_boundary_id);
        }
        else if (face_center[1] < radius &&
                  face_center[1] > -radius &&
                    face_center[2] > 0 && face_center[2] < depth)
        {
          active_cell->face(face_index)->set_all_boundary_ids(
            face_center[0] < 0.0 ?
              left_notch_boundary_id :
              right_notch_boundary_id);
        }
      }
    }
  }

  const dealii::Point<3>
    z_axis(0.,0.,1.),
    left_notch_center(-1.,0.,0.),
    right_notch_center(+1.,0.,0.);

  const dealii::CylindricalManifold<3>
    left_manifold(z_axis, left_notch_center),
    right_manifold(z_axis, right_notch_center);

  triangulation.set_all_manifold_ids_on_boundary(6, 6);

  triangulation.set_all_manifold_ids_on_boundary(7, 7);

  triangulation.set_manifold(6, left_manifold);

  triangulation.set_manifold(7, right_manifold);
}



void polycristalline_unit_square(
  dealii::parallel::distributed::Triangulation<2> &triangulation,
  const std::string msh_file_path)
{
  dealii::GridIn<2> grid_in;

  grid_in.attach_triangulation(triangulation);

  std::ifstream input_file(msh_file_path);

  grid_in.read_msh(input_file);

  const unsigned int
    x_lower_boundary_id = 0,
    x_upper_boundary_id = 1,
    y_lower_boundary_id = 2,
    y_upper_boundary_id = 3;

  // Identify boundaries
  for (const auto &active_cell : triangulation.active_cell_iterators())
  {
    if (active_cell->is_locally_owned() && active_cell->at_boundary())
    {
      for (const auto &face : active_cell->face_iterators())
      {
        if (face->at_boundary())
        {
          if (face->center()[0] == 0.)
            face->set_boundary_id(x_lower_boundary_id);
          else if (face->center()[0] == 1.)
            face->set_boundary_id(x_upper_boundary_id);
          else if (face->center()[1] == 0.)
            face->set_boundary_id(y_lower_boundary_id);
          else if (face->center()[1] == 1.)
            face->set_boundary_id(y_upper_boundary_id);
        }
      }
    }
  }
}



void polycristalline_unit_square(
  dealii::parallel::distributed::Triangulation<3> &triangulation,
  const std::string msh_file_path)
{
  dealii::parallel::distributed::Triangulation<2>
    internal_triangulation(
    MPI_COMM_WORLD,
    typename dealii::Triangulation<2>::MeshSmoothing(
    dealii::Triangulation<2>::smoothing_on_refinement |
    dealii::Triangulation<2>::smoothing_on_coarsening));

  dealii::GridIn<2> grid_in;

  grid_in.attach_triangulation(internal_triangulation);

  std::ifstream input_file(msh_file_path);

  grid_in.read_msh(input_file);

  const double depth =
    1./sqrt(internal_triangulation.n_global_active_cells());

  dealii::GridGenerator::extrude_triangulation(
    internal_triangulation,
    2,
    depth,
    triangulation);

  const unsigned int
    x_lower_boundary_id = 0,
    x_upper_boundary_id = 1,
    y_lower_boundary_id = 2,
    y_upper_boundary_id = 3,
    z_lower_boundary_id = 4,
    z_upper_boundary_id = 5;

  // Identify boundaries
  for (const auto &active_cell : triangulation.active_cell_iterators())
  {
    if (active_cell->is_locally_owned() && active_cell->at_boundary())
    {
      for (const auto &face : active_cell->face_iterators())
      {
        if (face->at_boundary())
        {
          if (face->center()[0] == 0.)
            face->set_boundary_id(x_lower_boundary_id);
          else if (face->center()[0] == 1.)
            face->set_boundary_id(x_upper_boundary_id);
          else if (face->center()[1] == 0.)
            face->set_boundary_id(y_lower_boundary_id);
          else if (face->center()[1] == 1.)
            face->set_boundary_id(y_upper_boundary_id);

          if (face->center()[2] == 0.)
            face->set_boundary_id(z_lower_boundary_id);
          else if (face->center()[2] == depth)
            face->set_boundary_id(z_upper_boundary_id);
        }
      }
    }
  }
}



} // namespace FiniteDomains



} // namespace gCP

