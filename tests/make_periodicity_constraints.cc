#include <deal.II/base/conditional_ostream.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_nothing.h>
#include <deal.II/fe/fe_system.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/hp/fe_collection.h>

#include <deal.II/numerics/data_out.h>

#include <string>

namespace Tests
{



template<int dim>
class MWE
{
public:

  MWE(const bool flag_refine_global);

  void run();

private:

  dealii::ConditionalOStream                    pcout;

  dealii::parallel::distributed::Triangulation<dim>
                                                triangulation;

  dealii::DoFHandler<dim>                       dof_handler;

  dealii::hp::FECollection<dim>                 fe_collection;

  dealii::Vector<float>                         active_fe_index;

  dealii::Vector<float>                         nth_active_fe_index;

  const double                                  height;

  const double                                  width;

  unsigned int                                  n_crystals;

  const unsigned int                            n_slips;

  const bool                                    flag_refine_global;

  void make_grid();

  void setup_dofs();

  void output();
};



template<int dim>
MWE<dim>::MWE(const bool flag_refine_global)
:
pcout(std::cout,
      dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0),
triangulation(MPI_COMM_WORLD,
              typename dealii::Triangulation<dim>::MeshSmoothing(
              dealii::Triangulation<dim>::smoothing_on_refinement |
              dealii::Triangulation<dim>::smoothing_on_coarsening)),
dof_handler(triangulation),
height(1.0),
width(1.0),
n_slips(2),
flag_refine_global(flag_refine_global)
{}



template<int dim>
void MWE<dim>::run()
{
  make_grid();

  setup_dofs();
}



template<int dim>
void MWE<dim>::make_grid()
{
  std::vector<unsigned int> repetitions(2, 20);

  dealii::GridGenerator::subdivided_hyper_rectangle(
    triangulation,
    repetitions,
    dealii::Point<dim>(0,0),
    dealii::Point<dim>(width, height),
    true);

  std::vector<dealii::GridTools::PeriodicFacePair<
    typename dealii::parallel::distributed::Triangulation<dim>::cell_iterator>>
    periodicity_vector;

  dealii::GridTools::collect_periodic_faces(triangulation,
                                            0,
                                            1,
                                            0,
                                            periodicity_vector);

  triangulation.add_periodicity(periodicity_vector);

  triangulation.refine_global(flag_refine_global);

  // Set material ids
  for (const auto &cell : triangulation.active_cell_iterators())
    if (cell->is_locally_owned())
    {
      if (std::fabs(cell->center()[1]) < height/2.0)
        cell->set_material_id(0);
      else
        cell->set_material_id(1);
    }

  this->pcout << "Triangulation:"
              << std::endl
              << " Number of active cells       = "
              << triangulation.n_global_active_cells()
              << std::endl << std::endl;
}



template<int dim>
void MWE<dim>::setup_dofs()
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

  // Set the active finite elemente index of each cell
  for (const auto &cell :
       dof_handler.active_cell_iterators())
    if (cell->is_locally_owned())
      cell->set_active_fe_index(cell->material_id());

  const unsigned int displacement_finite_element_degree = 2;
  const unsigned int slip_finite_element_degree         = 1;

  for (dealii::types::material_id i = 0; i < n_crystals; ++i)
  {
    std::vector<const dealii::FiniteElement<dim>*>  finite_elements;

    // A
    if (false)
      for (dealii::types::material_id j = 0; j < n_crystals; ++j)
        for (unsigned int k = 0; k < dim; ++k)
          if (i == j)
            finite_elements.push_back(
              new dealii::FE_Q<dim>(displacement_finite_element_degree));
          else
            finite_elements.push_back(
              new dealii::FE_Nothing<dim>());
    else
      for (unsigned int j = 0; j < dim; ++j)
        finite_elements.push_back(
          new dealii::FE_Q<dim>(displacement_finite_element_degree));

    // B
    for (dealii::types::material_id j = 0; j < n_crystals; ++j)
      for (unsigned int k = 0; k < n_slips; ++k)
        if (i == j)
          finite_elements.push_back(
            new dealii::FE_Q<dim>(slip_finite_element_degree));
        else
          finite_elements.push_back(
            new dealii::FE_Nothing<dim>());

    fe_collection.push_back(
      dealii::FESystem<dim>(
        finite_elements,
        std::vector<unsigned int>(finite_elements.size(), 1)));

    for (auto finite_element: finite_elements)
      delete finite_element;
    finite_elements.clear();
  }

  dof_handler.distribute_dofs(fe_collection);

  dealii::DoFRenumbering::Cuthill_McKee(dof_handler);

  output();

  dealii::IndexSet locally_owned_dofs = dof_handler.locally_owned_dofs();

  dealii::IndexSet locally_relevant_dofs;

  dealii::DoFTools::extract_locally_relevant_dofs(
    dof_handler,
    locally_relevant_dofs);

  std::vector<
    dealii::GridTools::
    PeriodicFacePair<typename dealii::DoFHandler<dim>::cell_iterator>>
      periodicity_vector;

  dealii::GridTools::collect_periodic_faces(
    dof_handler,
    0,
    1,
    0,
    periodicity_vector);

  dealii::AffineConstraints<double> hanging_node_constraints;

  hanging_node_constraints.clear();
  {
    hanging_node_constraints.reinit(locally_relevant_dofs);

    dealii::DoFTools::make_hanging_node_constraints(
      dof_handler,
      hanging_node_constraints);
  }
  hanging_node_constraints.close();

  dealii::AffineConstraints<double> affine_constraints;

  affine_constraints.clear();
  {
    affine_constraints.reinit(locally_relevant_dofs);
    affine_constraints.merge(hanging_node_constraints);

    dealii::DoFTools::make_periodicity_constraints<dim, dim>(
      periodicity_vector,
      affine_constraints);
  }
  affine_constraints.close();
}



template<int dim>
void MWE<dim>::output()
{
  active_fe_index.reinit(triangulation.n_active_cells());
  nth_active_fe_index.reinit(triangulation.n_active_cells());

  active_fe_index     = -1.0;
  nth_active_fe_index = -1.0;

  for (const auto &cell :
       dof_handler.active_cell_iterators())
    if (cell->is_locally_owned())
    {
      active_fe_index(cell->active_cell_index()) =
        cell->active_fe_index();

      if (cell->at_boundary())
        for (const auto &face_id : cell->face_indices())
          if (cell->face(face_id)->at_boundary())
          {
            AssertThrow(
              cell->face(face_id)->get_active_fe_indices().size() == 1,
              dealii::ExcMessage("Error!"));

            nth_active_fe_index(cell->active_cell_index()) =
              cell->face(face_id)->nth_active_fe_index(0);

            AssertThrow(cell->face(face_id)->fe_index_is_active(
              cell->face(face_id)->nth_active_fe_index(0)),
              dealii::ExcMessage("Error!"));

            cell->face(face_id)->get_fe(
              cell->face(face_id)->nth_active_fe_index(0));
          }
    }

  dealii::DataOut<dim> data_out;

  data_out.attach_dof_handler(dof_handler);

  data_out.add_data_vector(active_fe_index,
                           "active_fe_index");

  data_out.add_data_vector(nth_active_fe_index,
                           "nth_active_fe_index");

  data_out.build_patches();

  data_out.write_vtu_in_parallel(
   "triangulation.vtu",
    MPI_COMM_WORLD);
}



} // namespace Test



int main(int argc, char *argv[])
{
  try
  {
    dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(
      argc, argv, dealii::numbers::invalid_unsigned_int);

    bool flag_refine_global;

    switch (argc)
    {
      case 1:
        flag_refine_global = true;
        break;
      case 2:
        {
          const std::string arg(argv[1]);

          if (arg == "true")
            flag_refine_global = true;
          else if (arg == "false")
            flag_refine_global = false;
          else
            AssertThrow(false, dealii::ExcNotImplemented());
        }
        break;
      default:
        AssertThrow(false, dealii::ExcNotImplemented());
    }


    Tests::MWE<2> test(flag_refine_global);
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