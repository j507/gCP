#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/types.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>

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

  dealii::Vector<float>                         cell_is_at_interface;

  const double                                  edge_length;

  void make_grid();

  void update_ghost_cells_data(dealii::DoFHandler<dim> &dof_handler);

  void mark_grid();

  void output();
};



template<int dim>
MarkInterface<dim>::MarkInterface()
:
pcout(
  std::cout,
  dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0),
triangulation(MPI_COMM_WORLD),
dof_handler(triangulation),
edge_length(1.0)
{}



template<int dim>
void MarkInterface<dim>::run()
{
  make_grid();

  update_ghost_cells_data(dof_handler);

  mark_grid();

  output();
}



template<int dim>
void MarkInterface<dim>::make_grid()
{
  // Grid
  dealii::GridGenerator::hyper_cube(
    triangulation,
    0,
    edge_length,
    false);

  triangulation.refine_global(5);

  // Subdomains
  for (const auto &active_cell : triangulation.active_cell_iterators())
    if (active_cell->is_locally_owned())
    {
      if (std::fabs(active_cell->center()[0]) < edge_length/2.0)
        active_cell->set_material_id(1);
      else
        active_cell->set_material_id(2);
    }

  // Match active FE index to material index
  for (const auto &active_cell : dof_handler.active_cell_iterators())
    if (active_cell->is_locally_owned())
      active_cell->set_active_fe_index(active_cell->material_id());
}



template <int dim>
void MarkInterface<dim>::update_ghost_cells_data(dealii::DoFHandler<dim> &dof_handler)
{
  const unsigned int spacedim = dim;

  auto pack = [](
    const typename dealii::DoFHandler<dim, spacedim>::active_cell_iterator &cell)->
      typename dealii::DoFHandler<dim,dim>::active_fe_index_type
      {
        return cell->material_id();
      };

  auto unpack = [&dof_handler](
    const typename dealii::DoFHandler<dim,spacedim>::active_cell_iterator &cell,
    const typename dealii::DoFHandler<dim,dim>::active_fe_index_type      active_fe_index)->
      void
      {
        cell->set_material_id(active_fe_index);
      };

  dealii::GridTools::exchange_cell_data_to_ghosts<
    typename dealii::DoFHandler< dim, dim >::active_fe_index_type,
    dealii::DoFHandler<dim, spacedim>>(
      dof_handler,
      pack,
      unpack);
}



template <int dim>
void MarkInterface<dim>::mark_grid()
{
  cell_is_at_interface.reinit(triangulation.n_active_cells());
  locally_owned_subdomain.reinit(triangulation.n_active_cells());
  material_id.reinit(triangulation.n_active_cells());
  active_fe_index.reinit(triangulation.n_active_cells());

  cell_is_at_interface    = -1.0;
  locally_owned_subdomain = -1.0;
  material_id             = -1.0;
  active_fe_index         = -1.0;

  for (const auto &active_cell :
       dof_handler.active_cell_iterators())
  {
    if (active_cell->is_locally_owned())
    {
      locally_owned_subdomain(active_cell->active_cell_index()) =
        triangulation.locally_owned_subdomain();
      material_id(active_cell->active_cell_index()) =
        active_cell->material_id();
      active_fe_index(active_cell->active_cell_index()) =
        active_cell->active_fe_index();

      for (const auto &face_index : active_cell->face_indices())
      {
        if (!active_cell->face(face_index)->at_boundary() &&
            active_cell->material_id() !=
              active_cell->neighbor(face_index)->material_id())
        {
          cell_is_at_interface(active_cell->active_cell_index()) = 1.0;
          break;
        }
      }
    }
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

  data_out.add_data_vector(cell_is_at_interface,
                           "cell_is_at_interface");

  data_out.build_patches();

  data_out.write_vtu_in_parallel(
    "triangulation_" + std::to_string(dim) + ".vtu",
    MPI_COMM_WORLD);
}



} // namespace Test



int main(int argc, char *argv[])
{
  try
  {
    dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(
      argc, argv, dealii::numbers::invalid_unsigned_int);

    {
      Tests::MarkInterface<2> test;
      test.run();
    }

    {
      Tests::MarkInterface<3> test;
      test.run();
    }

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