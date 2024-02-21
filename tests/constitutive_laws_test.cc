#include <gCP/constitutive_laws.h>
#include <gCP/crystal_data.h>
#include <gCP/quadrature_point_history.h>
#include <gCP/run_time_parameters.h>
#include <gCP/utilities.h>

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

  CrystalData(
  const gCP::RunTimeParameters::BasicProblem &parameters);

  void run();

private:
  gCP::RunTimeParameters::BasicProblem               parameters;

  dealii::ConditionalOStream                              pcout;

  dealii::parallel::distributed::Triangulation<dim>       triangulation;

  const double                                            length;

  const double                                            height;

  const double                                            width;

  const unsigned int                                      string_width;

  std::vector<unsigned int>                               repetitions;

  std::shared_ptr<gCP::CrystalsData<dim>>                 crystals_data;

  gCP::Kinematics::ElasticStrain<dim>                     elastic_strain;

  gCP::ConstitutiveLaws::HookeLaw<dim>                    hooke_law;

  gCP::ConstitutiveLaws::ResolvedShearStressLaw<dim>      resolved_shear_stress_law;

  gCP::ConstitutiveLaws::ScalarMicrostressLaw<dim>        scalar_microstress_law;

  gCP::ConstitutiveLaws::VectorialMicrostressLaw<dim>     vectorial_microstress_law;

  gCP::ConstitutiveLaws::MicrotractionLaw<dim>      microtraction_law;

  gCP::ConstitutiveLaws::CohesiveLaw<dim>                 cohesive_law;

  gCP::ConstitutiveLaws::DegradationFunction              degradation_function;

  gCP::QuadraturePointHistory<dim>                        quadrature_point_history;

  gCP::InterfaceQuadraturePointHistory<dim>               interface_quadrature_point_history;

  void make_grid();

  void mark_grid();

  void init();

  void test_constitutive_laws();

  void output();
};



template<int dim>
CrystalData<dim>::CrystalData(
  const gCP::RunTimeParameters::BasicProblem &parameters)
:
parameters(parameters),
pcout(std::cout,
      dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0),
triangulation(MPI_COMM_WORLD,
              typename dealii::Triangulation<dim>::MeshSmoothing(
              dealii::Triangulation<dim>::smoothing_on_refinement |
              dealii::Triangulation<dim>::smoothing_on_coarsening)),
length(1.0),
height(1.0),
width(1.0),
string_width(32),
repetitions(dim, 10),
crystals_data(std::make_shared<gCP::CrystalsData<dim>>()),
elastic_strain(crystals_data),
hooke_law(
  crystals_data,
  parameters.solver_parameters.constitutive_laws_parameters.hooke_law_parameters),
resolved_shear_stress_law(crystals_data),
scalar_microstress_law(
  crystals_data,
  parameters.solver_parameters.constitutive_laws_parameters.scalar_microstress_law_parameters),
vectorial_microstress_law(
  crystals_data,
  parameters.solver_parameters.constitutive_laws_parameters.vectorial_microstress_law_parameters),
microtraction_law(
  crystals_data,
  parameters.solver_parameters.constitutive_laws_parameters.microtraction_law_parameters),
cohesive_law(parameters.solver_parameters.constitutive_laws_parameters.cohesive_law_parameters),
degradation_function(parameters.solver_parameters.constitutive_laws_parameters.degradation_function_parameters)
{
  this->pcout << "TESTING CONSTITUTIVE LAWS IN " << std::noshowpos
              << dim << "-D..." << std::endl << std::endl;
}



template<int dim>
void CrystalData<dim>::run()
{
  make_grid();

  mark_grid();

  init();

  test_constitutive_laws();
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
  for (const auto &cell : triangulation.active_cell_iterators())
    if (cell->is_locally_owned())
    {
      if (std::fabs(cell->center()[0]) < length/2.0)
        cell->set_material_id(0);
      else
        cell->set_material_id(1);
    }
}


template<int dim>
void CrystalData<dim>::init()
{
  crystals_data->init(triangulation,
                      parameters.input.euler_angles_pathname,
                      parameters.input.slips_directions_pathname,
                      parameters.input.slips_normals_pathname);

  this->pcout
    << "Overall data" << std::endl
    << std::setw(string_width) << std::left << " Number of crystals" << " = "
    << crystals_data->get_n_crystals()
    << std::endl
    << std::setw(string_width) << std::left << " Number of slip systems" << " = "
    << crystals_data->get_n_slips()
    << std::endl << std::endl;

  for (unsigned int crystal_id = 0;
       crystal_id < crystals_data->get_n_crystals(); ++crystal_id)
    for (unsigned int i = 0; i  < crystals_data->get_n_slips(); ++i)
      std::cout
        << "Crystal - " << crystal_id << " | Slip system - " << i << "\n"
        << std::setw(string_width) << std::left << " Direction" << " = "
        << gCP::Utilities::get_tensor_as_string(
            crystals_data->get_slip_direction(crystal_id,i)) << "\n\n"
        << std::setw(string_width) << std::left << " Normal" << " = "
        << gCP::Utilities::get_tensor_as_string(
            crystals_data->get_slip_normal(crystal_id,i)) << "\n\n"
        << std::setw(string_width) << std::left <<  " Schmid-Tensor" << " = "
        << gCP::Utilities::get_tensor_as_string(
            crystals_data->get_schmid_tensor(crystal_id,i), string_width + 3) << "\n\n"
        << std::setw(string_width) << " Symmetric Schmid-Tensor" << " = "
        << gCP::Utilities::get_tensor_as_string(
            crystals_data->get_symmetrized_schmid_tensor(crystal_id,i), string_width + 3)
        << "\n\n";

  hooke_law.init();

  vectorial_microstress_law.init();

  quadrature_point_history.init(
    parameters.solver_parameters.constitutive_laws_parameters.scalar_microstress_law_parameters,
    crystals_data->get_n_slips());

  interface_quadrature_point_history.init(
    parameters.solver_parameters.constitutive_laws_parameters.damage_evolution_parameters,
    parameters.solver_parameters.constitutive_laws_parameters.cohesive_law_parameters);
}



template<int dim>
void CrystalData<dim>::test_constitutive_laws()
{
  std::cout << "Testing ElasticStrain<dim> \n\n";

  dealii::SymmetricTensor<2,dim>  strain_tensor;

  strain_tensor[0][0] = 1.0;
  strain_tensor[1][1] = 1.0;

  std::vector<std::vector<double>> slip_values(
    crystals_data->get_n_slips(),
    std::vector<double>(1, 0 ));

  slip_values[0][0] = 0.5;
  slip_values[1][0] = 2.0;

  const dealii::SymmetricTensor<2,dim> elastic_strain_tensor =
    elastic_strain.get_elastic_strain_tensor(
      0, // crystal_id
      0, // q_point
      strain_tensor,
      slip_values);

  std::cout
    << std::setw(string_width) << std::left << " Strain tensor" << " = "
    << gCP::Utilities::get_tensor_as_string(strain_tensor, string_width + 3)
    << "\n\n";

  for (unsigned int i = 0; i  < crystals_data->get_n_slips(); ++i)
    std::cout
      << std::setw(string_width) << std::left << (" Slip " + std::to_string(i)) << " = "
      << std::fixed << std::left << std::showpos << std::setprecision(6)
      <<  slip_values[i][0] << "\n\n"
      << std::setw(string_width) << std::left << (" Plastic strain " + std::to_string(i)) << " = "
      << gCP::Utilities::get_tensor_as_string(
          slip_values[i][0] * crystals_data->get_symmetrized_schmid_tensor(0,i),string_width + 3) << "\n\n";

  std::cout
    << std::setw(string_width) << std::left << " Elastic strain tensor" << " = "
    << gCP::Utilities::get_tensor_as_string(elastic_strain_tensor, string_width + 3)
    << "\n\n";

  std::cout << "Testing HookeLaw<dim> \n\n";

  const dealii::SymmetricTensor<2,dim> stress_tensor =
    hooke_law.get_stress_tensor(0, // crystal_id
                               elastic_strain_tensor);

  const dealii::SymmetricTensor<4,dim> stiffness_tetrad =
    hooke_law.get_stiffness_tetrad(0);

  std::cout << std::setw(string_width) << std::left << " Stiffness tetrad" << " = "
            << gCP::Utilities::print_tetrad(stiffness_tetrad, string_width + 3, 15, 3, true)
            << "\n\n";

  std::cout << std::setw(string_width) << std::left << " Stress tensor" << " = "
            << gCP::Utilities::get_tensor_as_string(stress_tensor, string_width + 3, 15, 3, true)
            << "\n\n";

  std::cout << "Testing ResolvedShearStressLaw<dim> \n\n";

  for (unsigned int i = 0; i  < crystals_data->get_n_slips(); ++i)
    std::cout
      << std::setw(string_width) << std::left << (" Resolved shear stress " + std::to_string(i)) << " = "
      << std::fixed << std::left << std::showpos << std::setprecision(4)
      << std::scientific
      << resolved_shear_stress_law.get_resolved_shear_stress(
          0, //crystal_id
          i, //slip_id
          stress_tensor)
      << "\n\n";

  std::cout << "Testing QuadraturePointHistory<dim> \n\n";

  std::vector<std::vector<double>> old_slip_values(
    crystals_data->get_n_slips(),
    std::vector<double>(1, 0));

  old_slip_values[0][0] = 1.5;
  old_slip_values[1][0] = 0.75;

  quadrature_point_history.update_values(
    0, // q_point
    slip_values,
    old_slip_values);

  quadrature_point_history.store_current_values();

  std::vector<double> slip_resistances =
    quadrature_point_history.get_slip_resistances();

  for (unsigned int i = 0; i  < crystals_data->get_n_slips(); ++i)
    std::cout
      << std::setw(string_width) << std::left << (" Slip " + std::to_string(i)) << " = "
      << std::fixed << std::left << std::showpos << std::setprecision(6)
      <<  slip_values[i][0] << "\n\n"
      << std::setw(string_width) << std::left << (" Old slip " + std::to_string(i)) << " = "
      << std::fixed << std::left << std::showpos << std::setprecision(6)
      <<  old_slip_values[i][0] << "\n\n"
      << std::setw(string_width) << std::left << (" Slip resistance " + std::to_string(i)) << " = "
      << std::fixed << std::left << std::showpos << std::setprecision(6)
      <<  slip_resistances[i]
      << "\n\n";


  std::cout << "Testing ScalarMicrostressLaw<dim> \n\n";



  const double time_step_size = 1e-2;

  std::cout << std::setw(string_width) << std::left << " Time step size " << " = "
            << time_step_size << "\n\n";

  for (unsigned int i = 0; i  < crystals_data->get_n_slips(); ++i)
    std::cout
      << std::setw(string_width) << std::left << (" Slip " + std::to_string(i)) << " = "
      << std::fixed << std::left << std::showpos << std::setprecision(6)
      <<  slip_values[i][0] << "\n\n"
      << std::setw(string_width) << std::left << (" Old slip " + std::to_string(i)) << " = "
      << std::fixed << std::left << std::showpos << std::setprecision(6)
      <<  old_slip_values[i][0] << "\n\n"
      << std::setw(string_width) << std::left << (" Hardening " + std::to_string(i)) << " = "
      << std::fixed << std::left << std::showpos << std::setprecision(6)
      <<  slip_resistances[i] << "\n\n"
      << std::setw(string_width) << std::left << (" Scalar microscopic stress " + std::to_string(i)) << " = "
      << std::fixed << std::left << std::showpos << std::setprecision(4)
      << std::scientific
      << scalar_microstress_law.get_scalar_microstress(
          slip_values[i][0],
          old_slip_values[i][0],
          slip_resistances[i],
          time_step_size)
      << "\n\n";


  const dealii::FullMatrix<double> gateaux_derivative_matrix =
    scalar_microstress_law.get_jacobian(
      0, // q_point
      slip_values,
      old_slip_values,
      slip_resistances,
      time_step_size);

  std::cout
    << std::setw(string_width) << std::left
    << " Gateaux derivative" << " = "
    << gCP::Utilities::get_fullmatrix_as_string(
      gateaux_derivative_matrix, string_width + 3, 15, 3, true) << "\n\n";

  std::cout << "Testing VectorialMicrostressLaw<dim> \n\n";

  dealii::Tensor<1,dim> slip_gradient;

  slip_gradient[0] = 0.5;
  slip_gradient[1] = 1.5;

  std::vector<dealii::Tensor<1,dim>>
    vectorial_microstresses(crystals_data->get_n_slips());

  for (unsigned int slip_id = 0;
       slip_id < crystals_data->get_n_slips(); ++slip_id)
    vectorial_microstresses[slip_id] =
      vectorial_microstress_law.get_vectorial_microstress(
        0, // crystal_id
        slip_id,
        slip_gradient);

  std::cout
    << std::setw(string_width) << std::left
    << " Slip gradient" << " = "
    << gCP::Utilities::get_tensor_as_string(slip_gradient)
    << "\n\n";

  for (unsigned int slip_id = 0;
       slip_id < crystals_data->get_n_slips(); ++slip_id)
    std::cout
      << std::setw(string_width) << std::left
      << (" Vector microscopic stress - " + std::to_string(slip_id)) << " = "
      << gCP::Utilities::get_tensor_as_string(vectorial_microstresses[slip_id])
      << "\n\n";

  std::cout << "Testing MicrotractionLaw<dim> \n\n";

  std::vector<dealii::Tensor<1,dim>> normal_vector_values(1);

  normal_vector_values[0][1] = 1.0;

  auto grain_interaction_moduli =
    microtraction_law.get_grain_interaction_moduli(
      0,
      1,
      normal_vector_values);

  std::vector<std::vector<double>> microtraction_values(
    crystals_data->get_n_slips(),
    std::vector<double>(1, 0));

  std::vector<std::vector<double>> face_slip_values(
    crystals_data->get_n_slips(),
    std::vector<double>(1, 0));

  std::vector<std::vector<double>> neighbour_face_slip_values(
    crystals_data->get_n_slips(),
    std::vector<double>(1, 0));

  face_slip_values[0][0] = 1.5;
  face_slip_values[1][0] = 0.5;

  neighbour_face_slip_values[0][0] = 0.75;
  neighbour_face_slip_values[1][0] = 2.0;

  for (unsigned int slip_id = 0;
       slip_id < crystals_data->get_n_slips(); ++slip_id)
    microtraction_values[slip_id][0] =
      microtraction_law.get_microtraction(
        0,
        slip_id,
        grain_interaction_moduli,
        face_slip_values,
        neighbour_face_slip_values);

  const dealii::FullMatrix<double> intra_gateaux_derivative =
    microtraction_law.get_intra_gateaux_derivative(
      0,
      grain_interaction_moduli);

  const dealii::FullMatrix<double> inter_gateaux_derivative =
    microtraction_law.get_inter_gateaux_derivative(
      0,
      grain_interaction_moduli);

  for (unsigned int i = 0; i  < crystals_data->get_n_slips(); ++i)
    std::cout
      << std::setw(string_width) << std::left << (" Face slip " + std::to_string(i)) << " = "
      << std::fixed << std::left << std::showpos << std::setprecision(6)
      <<  face_slip_values[i][0] << "\n\n";

  for (unsigned int i = 0; i  < crystals_data->get_n_slips(); ++i)
    std::cout
      << std::setw(string_width) << std::left << (" Neighbour face slip " + std::to_string(i)) << " = "
      << std::fixed << std::left << std::showpos << std::setprecision(6)
      <<  neighbour_face_slip_values[i][0] << "\n\n";

  std::cout
    << std::setw(string_width) << std::left
    << " Intra grain interaction moduli" << " = "
    << gCP::Utilities::get_fullmatrix_as_string(
      grain_interaction_moduli.first[0], string_width + 3, 15, 3, true)
    << "\n\n"
    << std::setw(string_width) << std::left
    << " Inter grain interaction moduli" << " = "
    << gCP::Utilities::get_fullmatrix_as_string(
      grain_interaction_moduli.second[0], string_width + 3, 15, 3, true)
    << "\n\n";

  for (unsigned int slip_id = 0;
       slip_id < crystals_data->get_n_slips(); ++slip_id)
    std::cout
      << std::setw(string_width) << std::left
      << (" Microscopic traction - " + std::to_string(slip_id)) << " = "
      << microtraction_values[slip_id][0]
      << "\n\n";

  std::cout
    << std::setw(string_width) << std::left
    << " Intra gateaux derivative" << " = "
    << gCP::Utilities::get_fullmatrix_as_string(
       intra_gateaux_derivative, string_width + 3, 15, 3, true)
    << "\n\n"
    << std::setw(string_width) << std::left
    << " Inter gateaux derivative" << " = "
    << gCP::Utilities::get_fullmatrix_as_string(
       inter_gateaux_derivative, string_width + 3, 15, 3, true)
    << "\n\n";

  const double microtraction_free_energy_density =
    microtraction_law.get_free_energy_density(
            1,
            0,
            0,
            normal_vector_values,
            neighbour_face_slip_values,
            face_slip_values);

  std::cout
    << std::setw(string_width) << std::left
    << " Free energy density" << " = "
    << microtraction_free_energy_density
    << "\n\n";

  std::cout << "Testing CohesiveLaw<dim> \n\n";

  dealii::Tensor<1,dim> opening_displacement;
  opening_displacement[0] = .01;
  opening_displacement[1] = .00;

  const double effective_opening_displacement =
    cohesive_law.get_effective_opening_displacement(
        opening_displacement,
        normal_vector_values[0]);

  std::cout
    << std::setw(string_width) << std::left
    << " Effective opening displacement" << " = "
    << effective_opening_displacement
    << "\n\n";

  const double cohesive_law_free_energy_density =
    cohesive_law.get_free_energy_density(
        effective_opening_displacement);

  std::cout
    << std::setw(string_width) << std::left
    << " Free energy density" << " = "
    << cohesive_law_free_energy_density
    << "\n\n";


  std::cout << "Testing InterfaceQuadraturePointHistory<dim> \n\n";

  const double thermodynamic_force =
    - degradation_function.get_degradation_function_derivative_value(
        interface_quadrature_point_history.get_damage_variable(), true) *
    (cohesive_law_free_energy_density
      +
     microtraction_free_energy_density);

  const double old_damage_value =
    interface_quadrature_point_history.get_damage_variable();

  interface_quadrature_point_history.update_values(
    effective_opening_displacement,
    thermodynamic_force);

  const double damage_value =
    interface_quadrature_point_history.get_damage_variable();

  std::cout
    << std::setw(string_width) << std::left
    << " Old damage value" << " = "
    << old_damage_value
    << "\n\n"
    << std::setw(string_width) << std::left
    << " Damage value" << " = "
    << damage_value
    << "\n\n";

}



} // namespace Test




int main(int argc, char *argv[])
{
  try
  {
    dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(
      argc, argv, dealii::numbers::invalid_unsigned_int);

    {
      gCP::RunTimeParameters::BasicProblem parameters("input/2d.prm");

      Tests::CrystalData<2> test_2d(parameters);
      test_2d.run();
    }

    {
      gCP::RunTimeParameters::BasicProblem parameters("input/3d.prm");

      Tests::CrystalData<3> test_3d(parameters);
      test_3d.run();
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