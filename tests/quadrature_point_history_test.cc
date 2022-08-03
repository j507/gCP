#include <gCP/quadrature_point_history.h>

#include <deal.II/base/conditional_ostream.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/filtered_iterator.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>

#include <deal.II/base/quadrature_point_data.h>


#include <deal.II/numerics/data_out.h>

#include <string>

namespace Tests
{



template<int dim>
class QuadraturePointHistory
{
public:

  QuadraturePointHistory();

  void run();

private:

  dealii::ConditionalOStream                    pcout;

  dealii::parallel::distributed::Triangulation<dim>
                                                triangulation;

  const dealii::FE_Q<dim>                       fe_q;

  dealii::DoFHandler<dim>                       dof_handler;

  const dealii::QGauss<dim-1>                   face_quadrature_formula;

  dealii::Vector<float>                         locally_owned_subdomain;

  dealii::Vector<float>                         material_id;

  dealii::Vector<float>                         active_fe_index;

  dealii::Vector<float>                         cell_is_at_grain_boundary;

  const double                                  length;

  const double                                  height;

  const double                                  width;

  unsigned int                                  n_crystals;

  std::vector<unsigned int>                     repetitions;

  gCP::InterfaceDataStorage<
    typename dealii::Triangulation<dim>::cell_iterator,
    gCP::InterfaceData<dim>>       interface_data_storage;

  void make_grid();

  void mark_grid();

  void init_quadrature_point_history();

  void prepare_quadrature_point_history_for_update();

  void update_quadrature_point_history();

  void read_quadrature_point_history();

  void output();
};



template<int dim>
QuadraturePointHistory<dim>::QuadraturePointHistory()
:
pcout(std::cout,
      dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0),
triangulation(MPI_COMM_WORLD,
              typename dealii::Triangulation<dim>::MeshSmoothing(
              dealii::Triangulation<dim>::smoothing_on_refinement |
              dealii::Triangulation<dim>::smoothing_on_coarsening)),
fe_q(1),
dof_handler(triangulation),
face_quadrature_formula(3),
length(1.0),
height(2.0),
width(1.0),
repetitions(dim, 20)
{}



template<int dim>
void QuadraturePointHistory<dim>::run()
{
  make_grid();

  mark_grid();

  dof_handler.distribute_dofs(fe_q);

  init_quadrature_point_history();

  read_quadrature_point_history();

  prepare_quadrature_point_history_for_update();

  update_quadrature_point_history();

  read_quadrature_point_history();

  output();
}



template<int dim>
void QuadraturePointHistory<dim>::make_grid()
{
  repetitions[1] = 40;

  switch (dim)
  {
  case 2:
    dealii::GridGenerator::subdivided_hyper_rectangle(
      triangulation,
      repetitions,
      dealii::Point<dim>(0,0),
      dealii::Point<dim>(length,height),
      true);
    break;
  case 3:
    dealii::GridGenerator::subdivided_hyper_rectangle(
      triangulation,
      repetitions,
      dealii::Point<dim>(0,0,0),
      dealii::Point<dim>(length,height,width),
      true);
    break;
  default:
    Assert(false,
           dealii::ExcMessage("This test only runs in 2-D and 3-D"))
    break;
  }

  this->pcout << "Triangulation:"
              << std::endl
              << " Number of active cells       = "
              << triangulation.n_global_active_cells()
              << std::endl << std::endl;
}



template <int dim>
void QuadraturePointHistory<dim>::mark_grid()
{
  for (const auto &cell : triangulation.active_cell_iterators())
    if (cell->is_locally_owned())
    {
      if (std::fabs(cell->center()[1]) < height/2.0)
        cell->set_material_id(0);
      else
        cell->set_material_id(1);
    }

  {
    std::set<dealii::types::material_id> crystal_id_set;

    for (const auto &cell : triangulation.active_cell_iterators())
      if (cell->is_locally_owned())
        if (!crystal_id_set.count(cell->material_id()))
          crystal_id_set.emplace(cell->material_id());

    crystal_id_set =
      dealii::Utilities::MPI::compute_set_union(crystal_id_set,
                                                MPI_COMM_WORLD);

    n_crystals = crystal_id_set.size();

    this->pcout << "Gradient crystal plasticity: " << std::endl
                << " Number of crystals = " << n_crystals << std::endl
                << std::endl;
  }

  for (const auto &cell : dof_handler.active_cell_iterators())
    if (cell->is_locally_owned())
      cell->set_active_fe_index(cell->material_id());

  cell_is_at_grain_boundary.reinit(triangulation.n_active_cells());
  locally_owned_subdomain.reinit(triangulation.n_active_cells());
  material_id.reinit(triangulation.n_active_cells());
  active_fe_index.reinit(triangulation.n_active_cells());

  cell_is_at_grain_boundary = 0.0;
  locally_owned_subdomain   = 0.0;
  material_id               = 0.0;
  active_fe_index           = 0.0;

  for (const auto &cell :
       dof_handler.active_cell_iterators())
    if (cell->is_locally_owned())
    {
      locally_owned_subdomain(cell->active_cell_index()) =
        triangulation.locally_owned_subdomain();
      material_id(cell->active_cell_index()) =
        cell->material_id();
      active_fe_index(cell->active_cell_index()) =
        cell->active_fe_index();
    }

  for (const auto &cell : dof_handler.active_cell_iterators())
    if (cell->is_locally_owned())
      for (const auto &face_index : cell->face_indices())
        if (!cell->face(face_index)->at_boundary() &&
            cell->neighbor(face_index)->is_locally_owned() &&
            cell->material_id() !=
              cell->neighbor(face_index)->material_id())
        {
          cell_is_at_grain_boundary(cell->active_cell_index()) = 1.0;
          break;
        }

}



template<int dim>
void QuadraturePointHistory<dim>::init_quadrature_point_history()
{
  using CellFilter =
    dealii::FilteredIterator<
      typename dealii::DoFHandler<dim>::active_cell_iterator>;

  const unsigned int n_q_points =
    face_quadrature_formula.size();

  interface_data_storage.initialize(
    CellFilter(dealii::IteratorFilters::LocallyOwnedCell(),
               dof_handler.begin_active()),
    CellFilter(dealii::IteratorFilters::LocallyOwnedCell(),
               dof_handler.end()),
    n_q_points);

  for (const auto &cell : dof_handler.active_cell_iterators())
    if (cell->is_locally_owned() &&
        cell_is_at_grain_boundary(cell->active_cell_index()))
      for (const auto &face_index : cell->face_indices())
        if (!cell->face(face_index)->at_boundary() &&
            cell->material_id() !=
              cell->neighbor(face_index)->material_id())
        {
          const std::vector<std::shared_ptr<gCP::InterfaceData<dim>>>
            local_quadrature_point_history =
              interface_data_storage.get_data(
                cell->id(),
                cell->neighbor(face_index)->id());

          Assert(local_quadrature_point_history.size() == n_q_points,
                  dealii::ExcInternalError());

          for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
          {
            local_quadrature_point_history[q_point]->init(q_point/10.);
          }
        }
}



template<int dim>
void QuadraturePointHistory<dim>::prepare_quadrature_point_history_for_update()
{
  const unsigned int n_q_points = face_quadrature_formula.size();

  for (const auto &cell : dof_handler.active_cell_iterators())
    if (cell->is_locally_owned() &&
        cell_is_at_grain_boundary(cell->active_cell_index()))
      for (const auto &face_index : cell->face_indices())
        if (!cell->face(face_index)->at_boundary() &&
            cell->material_id() !=
              cell->neighbor(face_index)->material_id())
        {
          const std::vector<std::shared_ptr<gCP::InterfaceData<dim>>>
            local_quadrature_point_history =
              interface_data_storage.get_data(
                cell->id(),
                cell->neighbor(face_index)->id());

          Assert(local_quadrature_point_history.size() == n_q_points,
                  dealii::ExcInternalError());

          for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
            local_quadrature_point_history[q_point]->prepare_for_update_call();
        }
}



template<int dim>
void QuadraturePointHistory<dim>::update_quadrature_point_history()
{
  const unsigned int n_q_points = face_quadrature_formula.size();

  const dealii::UpdateFlags face_update_flags  =
    dealii::update_quadrature_points;

  dealii::FEFaceValues<dim>  fe_face_values(
                               fe_q,
                               face_quadrature_formula,
                               face_update_flags);

  for (const auto &cell : dof_handler.active_cell_iterators())
    if (cell->is_locally_owned() &&
        cell_is_at_grain_boundary(cell->active_cell_index()))
      for (const auto &face_index : cell->face_indices())
        if (!cell->face(face_index)->at_boundary() &&
            cell->material_id() !=
              cell->neighbor(face_index)->material_id())
        {
          fe_face_values.reinit(cell, face_index);

          const std::vector<dealii::Point<dim>> quadrature_points =
            fe_face_values.get_quadrature_points();

          const std::vector<std::shared_ptr<gCP::InterfaceData<dim>>>
            local_quadrature_point_history =
              interface_data_storage.get_data(
                cell->id(),
                cell->neighbor(face_index)->id());

          Assert(local_quadrature_point_history.size() == n_q_points,
                  dealii::ExcInternalError());

          for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
            local_quadrature_point_history[q_point]->update(
              quadrature_points[q_point],
              dealii::Tensor<1,dim>());
        }
}



template<int dim>
void QuadraturePointHistory<dim>::read_quadrature_point_history()
{
  const unsigned int n_q_points = face_quadrature_formula.size();

  for (const auto &cell : dof_handler.active_cell_iterators())
    if (cell->is_locally_owned() &&
        cell_is_at_grain_boundary(cell->active_cell_index()))
      for (const auto &face_index : cell->face_indices())
        if (!cell->face(face_index)->at_boundary() &&
            cell->material_id() !=
              cell->neighbor(face_index)->material_id())
        {
          const std::vector<std::shared_ptr<gCP::InterfaceData<dim>>>
            local_quadrature_point_history =
              interface_data_storage.get_data(
                cell->id(),
                cell->neighbor(face_index)->id());

          Assert(local_quadrature_point_history.size() == n_q_points,
                  dealii::ExcInternalError());

          const dealii::CellId current_cell_id =
            cell->id();

          const dealii::CellId neighbour_cell_id =
            cell->neighbor(face_index)->id();

          std::pair<dealii::CellId, dealii::CellId> cell_pair;

          if (current_cell_id < neighbour_cell_id)
            cell_pair =
              std::make_pair(current_cell_id, neighbour_cell_id);
          else
            cell_pair =
              std::make_pair(neighbour_cell_id, current_cell_id);

          /*std::cout
            << "Pair {" << cell_pair.first << ", " << cell_pair.second
            << "}\n";*/

          for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
            std::cout
              << " [" << q_point << "].get_values() = "
              << local_quadrature_point_history[q_point]->get_value()
              << "\n";
        }
}



template<int dim>
void QuadraturePointHistory<dim>::output()
{
  dealii::DataOut<dim> data_out;

  data_out.attach_dof_handler(dof_handler);

  data_out.add_data_vector(locally_owned_subdomain,
                           "locally_owned_subdomain");

  data_out.add_data_vector(material_id,
                           "material_id");

  data_out.add_data_vector(active_fe_index,
                           "active_fe_index");

  data_out.add_data_vector(cell_is_at_grain_boundary,
                           "cell_is_at_grain_boundary");

  data_out.build_patches();

  data_out.write_vtu_in_parallel("triangulation.vtu",
                                 MPI_COMM_WORLD);
}



} // namespace Test



int main(int argc, char *argv[])
{
  try
  {
    dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(
      argc, argv, dealii::numbers::invalid_unsigned_int);

    Tests::QuadraturePointHistory<2> test;
    test.run();

  }
  catch (std::exception &exc)
  {
    std::cerr << std::endl
              << std::endl
              << "----------------------------------------------------"
              << std::endl;
    std::cerr << "Exception on processing: " << std::endl
              << exc.what() << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------"
              << std::endl;
    return 1;
  }
  catch (...)
  {
    std::cerr << std::endl
              << std::endl
              << "----------------------------------------------------"
              << std::endl;
    std::cerr << "Unknown exception!" << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------"
              << std::endl;
    return 1;
  }
  return 0;
}