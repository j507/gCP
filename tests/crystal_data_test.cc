#include <gCP/crystal_data.h>

#include <deal.II/base/conditional_ostream.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_nothing.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>

#include <deal.II/numerics/data_out.h>

#include <string>

namespace Tests
{



template<int dim>
class CrystalData
{
public:

  CrystalData();

  void run();

private:

  dealii::ConditionalOStream                        pcout;

  dealii::parallel::distributed::Triangulation<dim> triangulation;

  dealii::DoFHandler<dim>                           dof_handler;

  const double                                      length;

  const double                                      height;

  const double                                      width;

  std::vector<unsigned int>                         repetitions;

  gCP::CrystalsData<dim>                            crystals_data;

  void make_grid();

  void mark_grid();

  void test_methods();

  void output();
};



template<int dim>
CrystalData<dim>::CrystalData()
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
repetitions(dim, 10)
{}



template<int dim>
void CrystalData<dim>::run()
{
  make_grid();

  dof_handler.distribute_dofs(dealii::FE_Nothing<dim>());

  mark_grid();

  test_methods();

  output();
}



template<int dim>
void CrystalData<dim>::make_grid()
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
    Assert(false, dealii::ExcMessage("This only runs in 2-D and 3-D"))
    break;
  }

  this->pcout << "Triangulation:"
              << std::endl
              << " Number of active cells       = "
              << triangulation.n_global_active_cells()
              << std::endl << std::endl;
}



template<int dim>
void CrystalData<dim>::mark_grid()
{
  for (const auto &cell : dof_handler.active_cell_iterators())
    if (cell->is_locally_owned())
    {
      if (std::fabs(cell->center()[0]) < length/2.0)
        cell->set_material_id(0);
      else
        cell->set_material_id(1);
    }
}



template<int dim>
void CrystalData<dim>::test_methods()
{
  crystals_data.init(triangulation,
                     "input/3d_euler_angles",
                     "input/3d_slip_directions",
                     "input/3d_slip_normals");

  this->pcout << "Overall data" << std::endl
              << " n_crystals  = " << crystals_data.get_n_crystals()
              << std::endl
              << " n_slips     = " << crystals_data.get_n_slips()
              << std::endl << std::endl;

  for (unsigned int crystal_id = 0;
       crystal_id < crystals_data.get_n_crystals();
       crystal_id++)
  {
    this->pcout << "Crystal - " << std::to_string(crystal_id)
                << std::endl << std::endl;

      this->pcout << " rotation_tensors["
                  << std::setw(2) << std::to_string(crystal_id)
                  << ","
                  << std::setw(2) << std::to_string(0)
                  << "]           = "
                  << std::fixed << std::setprecision(6) << std::showpos
                  << crystals_data.get_rotation_tensor(crystal_id)
                  << std::noshowpos <<  std::endl << std::endl;

    for (unsigned int slip_id = 0;
         slip_id < crystals_data.get_n_slips();
         slip_id++)
    {
      this->pcout << " Slip - " << std::to_string(slip_id)
                  << std::endl << std::endl;

      this->pcout << "  slip_directions["
                  << std::setw(2) << std::to_string(crystal_id)
                  << ","
                  << std::setw(2) << std::to_string(slip_id)
                  << "]            = "
                  << std::fixed << std::setprecision(6) << std::showpos
                  << crystals_data.get_slip_direction(crystal_id,
                                                      slip_id)
                  << std::noshowpos <<  std::endl;

      this->pcout << "  slip_normals["
                  << std::setw(2) << std::to_string(crystal_id)
                  << ","
                  << std::setw(2) << std::to_string(slip_id)
                  << "]               = "
                  << std::fixed << std::setprecision(6) << std::showpos
                  << crystals_data.get_slip_normal(crystal_id,
                                                   slip_id)
                  << std::noshowpos <<  std::endl;

      this->pcout << "  slip_orthogonals["
                  << std::setw(2) << std::to_string(crystal_id)
                  << ","
                  << std::setw(2) << std::to_string(slip_id)
                  << "]           = "
                  << std::fixed << std::setprecision(6) << std::showpos
                  << crystals_data.get_slip_orthogonal(crystal_id,
                                                       slip_id)
                  << std::noshowpos <<  std::endl;

      this->pcout << "  schmid_tensors["
                  << std::setw(2) << std::to_string(crystal_id)
                  << ","
                  << std::setw(2) << std::to_string(slip_id)
                  << "]             = "
                  << std::fixed << std::setprecision(6) << std::showpos
                  << crystals_data.get_schmid_tensor(crystal_id, slip_id)
                  << std::noshowpos <<  std::endl;

      this->pcout << "  symmetrized_schmid_tensors["
                  << std::setw(2) << std::to_string(crystal_id)
                  << ","
                  << std::setw(2) << std::to_string(slip_id)
                  << "] = "
                  << std::fixed << std::setprecision(6) << std::showpos
                  << crystals_data.get_symmetrized_schmid_tensor(crystal_id, slip_id)
                  << std::noshowpos <<  std::endl;

      this->pcout << std::endl;
    }
    this->pcout << std::endl;
  }
}


template<int dim>
void CrystalData<dim>::output()
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

  data_out.add_data_vector(material_id_per_cell, "CrystalData");

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

    Tests::CrystalData<2> problem_2d;
    //problem_2d.run();

    Tests::CrystalData<3> problem_3d;
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