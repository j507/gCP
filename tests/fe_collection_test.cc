#include <deal.II/base/conditional_ostream.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_nothing.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values_extractors.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>

#include <deal.II/hp/fe_collection.h>
#include <deal.II/hp/fe_values.h>

#include <deal.II/numerics/data_out.h>

#include <string>

namespace Tests
{



template<int dim>
class FE_Collection
{
public:

  FE_Collection();

  void run();

private:

  dealii::ConditionalOStream                        pcout;

  dealii::parallel::distributed::Triangulation<dim> triangulation;

  dealii::DoFHandler<dim>                           dof_handler;

  dealii::hp::FECollection<dim>                     fe_collection;

  const double                                      length;

  const double                                      height;

  const double                                      width;

  unsigned int                                      n_crystals;

  const unsigned int                                n_slips;

  std::vector<unsigned int>                         repetitions;

  std::vector<dealii::FEValuesExtractors::Vector>   displacement_extractors;

  std::vector<std::vector<dealii::FEValuesExtractors::Scalar>>
                                                    slips_extractors;

  std::vector<unsigned int>                         dof_mapping;

  void make_grid();

  void setup();

  void output();
};



template<int dim>
FE_Collection<dim>::FE_Collection()
:
pcout(std::cout,
      dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0),
triangulation(MPI_COMM_WORLD,
              typename dealii::Triangulation<dim>::MeshSmoothing(
              dealii::Triangulation<dim>::smoothing_on_refinement |
              dealii::Triangulation<dim>::smoothing_on_coarsening)),
dof_handler(triangulation),
length(10.0),
height(1.0),
width(1.0),
n_slips(3),
repetitions(dim, 10)
{}



template<int dim>
void FE_Collection<dim>::run()
{
  make_grid();

  setup();

  output();
}



template<int dim>
void FE_Collection<dim>::make_grid()
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
    Assert(false, dealii::ExcMessage("This test only runs in 2-D and 3-D"))
    break;
  }

  this->pcout << "Triangulation:"
              << std::endl
              << " Number of active cells       = "
              << triangulation.n_global_active_cells()
              << std::endl << std::endl;
}



template<int dim>
void FE_Collection<dim>::setup()
{
  // Set material ids
  for (const auto &cell : triangulation.active_cell_iterators())
    if (cell->is_locally_owned())
    {
      if (std::fabs(cell->center()[0]) < length/2.0)
        cell->set_material_id(0);
      else
        cell->set_material_id(1);
    }


  // Get number of crystals in the triangulation
  // Here material_id is being misused to assign identifiers to the
  // different crystals
  {
    std::set<dealii::types::material_id> crystal_id_set;

    for (const auto &cell : triangulation.active_cell_iterators())
      if (cell->is_locally_owned())
        if (!crystal_id_set.count(cell->material_id()))
          crystal_id_set.emplace(cell->material_id());

    n_crystals = crystal_id_set.size();

    this->pcout << "Gradient crystal plasticity: " << std::endl
                << " Number of crystals = " << n_crystals << std::endl
                << " Number of slips    = " << n_slips << std::endl
                << std::endl;
  }

  // The finite element collection contains the finite element systems
  // corresponding to each crystal
  for (const auto &cell : dof_handler.active_cell_iterators())
    if (cell->is_locally_owned())
      cell->set_active_fe_index(cell->material_id());


  const unsigned int displacement_finite_element_degree = 1;
  const unsigned int slip_finite_element_degree         = 1;

  // The FESystem of the i-th crystal is divided into [ A | B ] with
  // dimensions [ dim x n_crystals | n_slips x n_crystals ] where
  //  A = FE_Nothing^dim     ... FE_Q^dim_i       ... FE_Nothing^dim
  //  B = FE_Nothing^n_slips ... FE_Q^{n_slips}_i ... FE_Nothing^n_slips
  // If the displacment is continuous across crystalls then [ A | B ] has
  // the dimensiones [ dim | n_slips x n_crystals ] where A = FE_Q^dim
  for (dealii::types::material_id i = 0; i < n_crystals; ++i)
  {
    std::vector<const dealii::FiniteElement<dim>*>  finite_elements;

    // A
    if (true)
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

    // Add [ A | B ] of the i-th crystal to the FECollection
    fe_collection.push_back(
      dealii::FESystem<dim>(
        finite_elements,
        std::vector<unsigned int>(finite_elements.size(), 1)));

    // Delete in order to avoid memory leaks
    for (auto finite_element: finite_elements)
      delete finite_element;
    finite_elements.clear();
  }

  this->pcout << "FECollection:" << std::endl;
  // Print all FESystem inside the FECollection to the terminal
  for (unsigned int j = 0; j < fe_collection.size(); ++j)
    this->pcout << " " << fe_collection[j].get_name() << std::endl;
  this->pcout << std::endl;

  std::cout << "hp::FEColleciont<dim>::max_dofs_per_cell() = "
            << fe_collection.max_dofs_per_cell()
            << std::endl;

  // displacement_extractors contains Vector extractors which can be
  // thought as the ComponentMasks of size n_crystals x (n_slips + dim)
  // [true  x dim  false x (n_crystals - 1) false x n_crystals x n_slips]
  // [false x dim  true x dim               false x (n_crystals - 2) ...]
  // and so on
  // slip_extractors_per_crystal contains for the first crystal
  // [false x dim x n_crystals  true   false  false ...]
  // [false x dim x n_crystals  false  true   false ...]
  // [false x dim x n_crystals  false  false  true  ...]
  // for the second crystal
  // [false x dim x n_crystals  false x n_slips  true   false  false ...]
  // [false x dim x n_crystals  false x n_slips  false  true   false ...]
  // [false x dim x n_crystals  false x n_slips  false  false  true  ...]
  // and so on
  // This is just a visual aid. They are not a vector of booleans!
  for (dealii::types::material_id i = 0; i < n_crystals; ++i)
  {
    displacement_extractors.push_back(dealii::FEValuesExtractors::Vector(i*dim));

    std::vector<dealii::FEValuesExtractors::Scalar> slip_extractors_per_crystal;

    for (unsigned int j = 0; j < n_slips; ++j)
      slip_extractors_per_crystal.push_back(
        dealii::FEValuesExtractors::Scalar(n_crystals * dim + i * n_slips + j));

    slips_extractors.push_back(slip_extractors_per_crystal);
  }



  dof_mapping.resize(n_crystals * (dim + n_slips));

  for (dealii::types::material_id i = 0; i < n_crystals; ++i)
  {
    for (unsigned int j = 0; j < dim; ++j)
      dof_mapping[i * dim + j] = j;

    for (unsigned int j = 0; j < n_slips; ++j)
      dof_mapping[n_crystals * dim + i * n_slips + j] = dim + j;
  }

  this->pcout << "dof_mapping = [";
  for (auto &value : dof_mapping)
    this->pcout << value << " ";
  this->pcout << "\b]" << std::endl << std::endl;
}



template<int dim>
void FE_Collection<dim>::output()
{
  dealii::DataOut<dim> data_out;

  data_out.attach_dof_handler(dof_handler);

  dealii::Vector<float> subdomain_per_cell(triangulation.n_active_cells());

  for (unsigned int i = 0; i < subdomain_per_cell.size(); ++i)
    subdomain_per_cell(i) = triangulation.locally_owned_subdomain();

  dealii::Vector<float> material_id_per_cell(triangulation.n_active_cells());

  for (const auto &cell : triangulation.active_cell_iterators())
    if (cell->is_locally_owned())
      material_id_per_cell(cell->active_cell_index()) = cell->material_id();

  data_out.add_data_vector(subdomain_per_cell, "Subdomain");

  data_out.add_data_vector(material_id_per_cell, "MaterialID");

  data_out.build_patches();

  static int out_index = 0;

  data_out.write_vtu_with_pvtu_record(
    "./",
    "solution_" + std::to_string(dim),
    out_index,
    MPI_COMM_WORLD,
    5);

  out_index++;
}



} // namespace Test




int main(int argc, char *argv[])
{
  try
  {
    dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(
      argc, argv, dealii::numbers::invalid_unsigned_int);

    Tests::FE_Collection<2> problem_2d;
    problem_2d.run();

    Tests::FE_Collection<3> problem_3d;
    problem_3d.run();
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