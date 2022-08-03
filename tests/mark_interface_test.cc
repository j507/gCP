#include <deal.II/base/conditional_ostream.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>

#include <deal.II/numerics/data_out.h>



namespace Tests
{



template<int dim>
class MarkInterface
{
public:

  MarkInterface();

  void run();

private:

  dealii::ConditionalOStream                    pcout;

  dealii::parallel::distributed::Triangulation<dim>
                                                triangulation;

  dealii::DoFHandler<dim>                       dof_handler;

  dealii::Vector<float>                         locally_owned_subdomain;

  dealii::Vector<float>                         material_id;

  dealii::Vector<float>                         active_fe_index;

  dealii::Vector<float>                         cell_is_at_grain_boundary;

  //dealii::LinearAlgebraTrilinos::MPI::Vector    cell_is_at_grain_boundary;

  const double                                  length;

  const double                                  height;

  const double                                  width;

  std::vector<unsigned int>                     repetitions;



  void make_grid();

  void mark_grid();

  void setup();

  void output();
};



template<int dim>
MarkInterface<dim>::MarkInterface()
:
pcout(std::cout,
      dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0),
triangulation(MPI_COMM_WORLD),
dof_handler(triangulation),
length(1.0),
height(1.0),
width(1.0),
repetitions(dim, 10)
{}



template<int dim>
void MarkInterface<dim>::run()
{
  make_grid();

  mark_grid();

  output();
}



template<int dim>
void MarkInterface<dim>::make_grid()
{
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
void MarkInterface<dim>::mark_grid()
{
  for (const auto &cell : triangulation.active_cell_iterators())
    if (cell->is_locally_owned())
    {
      if (std::fabs(cell->center()[0]) < length/2.0)
        cell->set_material_id(0);
      else
        cell->set_material_id(1);
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
      for (const auto &face_id : cell->face_indices())
        if (!cell->face(face_id)->at_boundary() &&
            cell->neighbor(face_id)->is_locally_owned() &&
            cell->material_id() !=
              cell->neighbor(face_id)->material_id())
        {
          cell_is_at_grain_boundary(cell->active_cell_index()) = 1.0;
          break;
        }

      locally_owned_subdomain(cell->active_cell_index()) =
        triangulation.locally_owned_subdomain();
      material_id(cell->active_cell_index()) =
        cell->material_id();
      active_fe_index(cell->active_cell_index()) =
        cell->active_fe_index();
    }
}



template<int dim>
void MarkInterface<dim>::output()
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

    Tests::MarkInterface<2> test;
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