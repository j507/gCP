#include <gCP/postprocessing.h>

#include <deal.II/base/work_stream.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/grid/filtered_iterator.h>

namespace gCP
{



namespace Postprocessing
{



template <int dim>
Postprocessor<dim>::Postprocessor(
  std::shared_ptr<FEField<dim>> &fe_field,
  std::shared_ptr<CrystalsData<dim>> &crystals_data,
  const RunTimeParameters::DimensionlessForm &parameters,
  const bool flag_light_output,
  const bool flag_output_dimensionless_quantities,
  const bool flag_output_fluctuations)
:
fe_field(fe_field),
crystals_data(crystals_data),
voigt_indices(6),
deviatoric_projector(
  dealii::identity_tensor<dim>() -
  1.0 / 3.0 *
  dealii::outer_product(
    dealii::unit_symmetric_tensor<dim>(),
    dealii::unit_symmetric_tensor<dim>())),
deviatoric_projector_3d(
  dealii::identity_tensor<3>() -
  1.0 / 3.0 *
  dealii::outer_product(
    dealii::unit_symmetric_tensor<3>(),
    dealii::unit_symmetric_tensor<3>())),
parameters(parameters),
flag_light_output(flag_light_output),
flag_output_dimensionless_quantities(
  flag_output_dimensionless_quantities),
flag_output_fluctuations(flag_output_fluctuations)
{
  macroscopic_strain = 0.;

  voigt_indices[0] = std::make_pair<unsigned int, unsigned int>(0,0);
  voigt_indices[1] = std::make_pair<unsigned int, unsigned int>(1,1);
  voigt_indices[2] = std::make_pair<unsigned int, unsigned int>(2,2);
  voigt_indices[3] = std::make_pair<unsigned int, unsigned int>(1,2);
  voigt_indices[4] = std::make_pair<unsigned int, unsigned int>(0,2);
  voigt_indices[5] = std::make_pair<unsigned int, unsigned int>(0,1);

  if (flag_output_dimensionless_quantities)
  {
    this->parameters.characteristic_quantities =
      RunTimeParameters::CharacteristicQuantities();
  }

  if (flag_light_output)
  {
    flag_init_was_called = true;
  }
}



template <int dim>
std::vector<std::string>
Postprocessor<dim>::get_names() const
{
  std::vector<std::string> solution_names(dim, "Displacement");

  for (unsigned int slip_id = 0;
      slip_id < crystals_data->get_n_slips();
      ++slip_id)
  {
    std::ostringstream osss;

    osss << std::setw(2) << std::setfill('0') << (slip_id + 1);

    solution_names.emplace_back("Slip_" + osss.str());
  }

  if (!flag_light_output)
  {
    solution_names.emplace_back("EquivalentPlasticStrain");
    solution_names.emplace_back("EquivalentAbsolutePlasticStrain");
    solution_names.emplace_back("EquivalentEdgeDislocationDensity");
    solution_names.emplace_back("EquivalentScrewDislocationDensity");
    solution_names.emplace_back("VonMisesStress");
    solution_names.emplace_back("VonMisesPlasticStrain");
    solution_names.emplace_back("Stress_11");
    solution_names.emplace_back("Stress_22");
    solution_names.emplace_back("Stress_33");
    solution_names.emplace_back("Stress_23");
    solution_names.emplace_back("Stress_13");
    solution_names.emplace_back("Stress_12");
    solution_names.emplace_back("Strain_11");
    solution_names.emplace_back("Strain_22");
    solution_names.emplace_back("Strain_33");
    solution_names.emplace_back("Strain_23x2");
    solution_names.emplace_back("Strain_13x2");
    solution_names.emplace_back("Strain_12x2");
    solution_names.emplace_back("ElasticStrain_11");
    solution_names.emplace_back("ElasticStrain_22");
    solution_names.emplace_back("ElasticStrain_33");
    solution_names.emplace_back("ElasticStrain_23x2");
    solution_names.emplace_back("ElasticStrain_13x2");
    solution_names.emplace_back("ElasticStrain_12x2");
    solution_names.emplace_back("PlasticStrain_11");
    solution_names.emplace_back("PlasticStrain_22");
    solution_names.emplace_back("PlasticStrain_33");
    solution_names.emplace_back("PlasticStrain_23x2");
    solution_names.emplace_back("PlasticStrain_13x2");
    solution_names.emplace_back("PlasticStrain_12x2");

    if (flag_output_fluctuations)
    {
      solution_names.emplace_back("VonMisesStressFluctuations");
      solution_names.emplace_back("StressFluctuations_11");
      solution_names.emplace_back("StressFluctuations_22");
      solution_names.emplace_back("StressFluctuations_33");
      solution_names.emplace_back("StressFluctuations_23");
      solution_names.emplace_back("StressFluctuations_13");
      solution_names.emplace_back("StressFluctuations_12");
      solution_names.emplace_back("StrainFluctuations_11");
      solution_names.emplace_back("StrainFluctuations_22");
      solution_names.emplace_back("StrainFluctuations_33");
      solution_names.emplace_back("StrainFluctuations_23x2");
      solution_names.emplace_back("StrainFluctuations_13x2");
      solution_names.emplace_back("StrainFluctuations_12x2");
    }
  }

  return solution_names;
}


template <int dim>
std::vector<dealii::DataComponentInterpretation::DataComponentInterpretation>
Postprocessor<dim>::get_data_component_interpretation()
  const
{
  std::vector<
    dealii::DataComponentInterpretation::DataComponentInterpretation>
      interpretation(
        dim,
        dealii::DataComponentInterpretation::component_is_part_of_vector);

  for (unsigned int slip_id = 0;
      slip_id < crystals_data->get_n_slips();
      ++slip_id)
    interpretation.push_back(
      dealii::DataComponentInterpretation::component_is_scalar);

  if (!flag_light_output)
  {
    interpretation.push_back(dealii::DataComponentInterpretation::component_is_scalar);
    interpretation.push_back(dealii::DataComponentInterpretation::component_is_scalar);
    interpretation.push_back(dealii::DataComponentInterpretation::component_is_scalar);
    interpretation.push_back(dealii::DataComponentInterpretation::component_is_scalar);
    interpretation.push_back(dealii::DataComponentInterpretation::component_is_scalar);
    interpretation.push_back(dealii::DataComponentInterpretation::component_is_scalar);
    interpretation.push_back(dealii::DataComponentInterpretation::component_is_scalar);
    interpretation.push_back(dealii::DataComponentInterpretation::component_is_scalar);
    interpretation.push_back(dealii::DataComponentInterpretation::component_is_scalar);
    interpretation.push_back(dealii::DataComponentInterpretation::component_is_scalar);
    interpretation.push_back(dealii::DataComponentInterpretation::component_is_scalar);
    interpretation.push_back(dealii::DataComponentInterpretation::component_is_scalar);
    interpretation.push_back(dealii::DataComponentInterpretation::component_is_scalar);
    interpretation.push_back(dealii::DataComponentInterpretation::component_is_scalar);
    interpretation.push_back(dealii::DataComponentInterpretation::component_is_scalar);
    interpretation.push_back(dealii::DataComponentInterpretation::component_is_scalar);
    interpretation.push_back(dealii::DataComponentInterpretation::component_is_scalar);
    interpretation.push_back(dealii::DataComponentInterpretation::component_is_scalar);

    // Elastic strain
    interpretation.push_back(dealii::DataComponentInterpretation::component_is_scalar);
    interpretation.push_back(dealii::DataComponentInterpretation::component_is_scalar);
    interpretation.push_back(dealii::DataComponentInterpretation::component_is_scalar);
    interpretation.push_back(dealii::DataComponentInterpretation::component_is_scalar);
    interpretation.push_back(dealii::DataComponentInterpretation::component_is_scalar);
    interpretation.push_back(dealii::DataComponentInterpretation::component_is_scalar);
    // Plastic strain
    interpretation.push_back(dealii::DataComponentInterpretation::component_is_scalar);
    interpretation.push_back(dealii::DataComponentInterpretation::component_is_scalar);
    interpretation.push_back(dealii::DataComponentInterpretation::component_is_scalar);
    interpretation.push_back(dealii::DataComponentInterpretation::component_is_scalar);
    interpretation.push_back(dealii::DataComponentInterpretation::component_is_scalar);
    interpretation.push_back(dealii::DataComponentInterpretation::component_is_scalar);

    if (flag_output_fluctuations)
    {
      interpretation.push_back(dealii::DataComponentInterpretation::component_is_scalar);
      interpretation.push_back(dealii::DataComponentInterpretation::component_is_scalar);
      interpretation.push_back(dealii::DataComponentInterpretation::component_is_scalar);
      interpretation.push_back(dealii::DataComponentInterpretation::component_is_scalar);
      interpretation.push_back(dealii::DataComponentInterpretation::component_is_scalar);
      interpretation.push_back(dealii::DataComponentInterpretation::component_is_scalar);
      interpretation.push_back(dealii::DataComponentInterpretation::component_is_scalar);
      interpretation.push_back(dealii::DataComponentInterpretation::component_is_scalar);
      interpretation.push_back(dealii::DataComponentInterpretation::component_is_scalar);
      interpretation.push_back(dealii::DataComponentInterpretation::component_is_scalar);
      interpretation.push_back(dealii::DataComponentInterpretation::component_is_scalar);
      interpretation.push_back(dealii::DataComponentInterpretation::component_is_scalar);
      interpretation.push_back(dealii::DataComponentInterpretation::component_is_scalar);
    }
  }

  return interpretation;
}


template <int dim>
dealii::UpdateFlags
Postprocessor<dim>::get_needed_update_flags() const
{
  return dealii::update_values |
         dealii::update_gradients |
         dealii::update_quadrature_points;
}



template <int dim>
void Postprocessor<dim>::init(
  std::shared_ptr<const ConstitutiveLaws::HookeLaw<dim>>  hooke_law)
{
  this->hooke_law       = hooke_law;

  flag_init_was_called  = true;
}



template <int dim>
void Postprocessor<dim>::set_macroscopic_strain(
  const dealii::SymmetricTensor<2,dim> macroscopic_strain)
{
  for (unsigned int i = 0;
        i < macroscopic_strain.n_independent_components;
        ++i)
    AssertIsFinite(macroscopic_strain.access_raw_entry(i));

  this->macroscopic_strain = macroscopic_strain;
}



template <int dim>
void Postprocessor<dim>::evaluate_vector_field(
  const dealii::DataPostprocessorInputs::Vector<dim>  &inputs,
  std::vector<dealii::Vector<double>>                 &computed_quantities) const
{
  AssertThrow(flag_init_was_called,
              dealii::ExcMessage("The Postprocessor<dim> instance has"
                                  " not been initialized."));

  // Get data
  const typename dealii::DoFHandler<dim>::cell_iterator current_cell =
    inputs.template get_cell<dim>();
  const unsigned int material_id = current_cell->material_id();
  const unsigned int n_quadrature_points =
    inputs.solution_values.size();
  const unsigned int n_components = fe_field->get_n_components();
  const unsigned int n_slips = crystals_data->get_n_slips();
  const unsigned int n_crystals = crystals_data->get_n_crystals();
  const double &dimensionless_number =
    parameters.dimensionless_numbers[0];

  (void)n_components;

  Assert(inputs.solution_gradients.size() == n_quadrature_points,
          dealii::ExcInternalError());
  Assert(computed_quantities.size() == n_quadrature_points,
          dealii::ExcInternalError());
  Assert(inputs.solution_values[0].size() == n_components,
          dealii::ExcInternalError());

  // Reset
  for (unsigned int quadrature_point_id = 0;
        quadrature_point_id < n_quadrature_points;
          ++quadrature_point_id)
  {
    for (unsigned int index = 0; index < computed_quantities[0].size();
          ++index)
    {
      computed_quantities[quadrature_point_id](index) = 0.0;
    }
  }

  // Local instances
  dealii::Tensor<2,dim> displacement_gradient;
  dealii::SymmetricTensor<2,dim> strain_tensor;
  dealii::SymmetricTensor<2,dim> elastic_strain_tensor;
  dealii::SymmetricTensor<2,dim> plastic_strain_tensor;
  dealii::SymmetricTensor<2,3> strain_tensor_3d;
  dealii::SymmetricTensor<2,3> plastic_strain_tensor_3d;
  dealii::SymmetricTensor<2,3> elastic_strain_tensor_3d;
  dealii::SymmetricTensor<2,3> stress_tensor;
  double equivalent_edge_dislocation_density;
  double equivalent_screw_dislocation_density;

  const unsigned int index_offset =
    fe_field->is_decohesion_allowed() ? dim * n_crystals : dim;

  for (unsigned int quadrature_point_index = 0;
        quadrature_point_index < n_quadrature_points;
          ++quadrature_point_index)
  {
    // Reset
    displacement_gradient                 = 0.0;
    strain_tensor                         = 0.0;
    strain_tensor_3d                      = 0.0;
    plastic_strain_tensor                 = 0.0;
    plastic_strain_tensor_3d              = 0.0;
    elastic_strain_tensor                 = 0.0;
    elastic_strain_tensor_3d              = 0.0;
    stress_tensor                         = 0.0;
    equivalent_edge_dislocation_density   = 0.0;
    equivalent_screw_dislocation_density  = 0.0;

    // Displacement
    for (unsigned int index = 0; index < dim; ++index)
    {
      if (fe_field->is_decohesion_allowed())
      {
        for (unsigned int crystal_id = 0;
              crystal_id < n_crystals; ++crystal_id)
        {
          computed_quantities[quadrature_point_index](index) +=
              parameters.characteristic_quantities.displacement *
              inputs.solution_values[quadrature_point_index](
                index + dim * crystal_id);

          if (!flag_light_output)
          {
            displacement_gradient[index] +=
              inputs.solution_gradients[quadrature_point_index][
                index + dim * crystal_id];
          }
        }
      }
      else
      {
        computed_quantities[quadrature_point_index](index) =
          parameters.characteristic_quantities.displacement *
          inputs.solution_values[quadrature_point_index](index);

        if (!flag_light_output)
        {
          displacement_gradient[index] =
            inputs.solution_gradients[quadrature_point_index][index];
        }
      }
    }

    // Slips
    for (unsigned int slip_id = 0;
        slip_id < n_slips; ++slip_id)
    {
      for (unsigned int crystal_id = 0;
          crystal_id < n_crystals; ++crystal_id)
      {
          // Slips
          computed_quantities[quadrature_point_index](dim + slip_id) +=
            inputs.solution_values[quadrature_point_index](
              index_offset + slip_id + n_slips * crystal_id);

          if (!flag_light_output)
          {
            // Equivalent plastic strain
            computed_quantities[quadrature_point_index](dim + n_slips) +=
                inputs.solution_values[quadrature_point_index](
                  index_offset + slip_id + n_slips * crystal_id);

            // Equivalent absolute plastic strain
            computed_quantities[quadrature_point_index](
              dim + n_slips + 1) +=
                std::abs(inputs.solution_values[quadrature_point_index](
                  index_offset + slip_id + n_slips * crystal_id));

            // Equivalent edge dislocation density
            equivalent_edge_dislocation_density +=
              std::pow(inputs.solution_gradients[quadrature_point_index][
                  index_offset + slip_id + n_slips * crystal_id] *
              crystals_data->get_slip_direction(crystal_id, slip_id), 2);

            // Equivalent screw dislocation density
            equivalent_screw_dislocation_density +=
              std::pow(inputs.solution_gradients[quadrature_point_index][
                  index_offset + slip_id + n_slips * crystal_id] *
              crystals_data->get_slip_orthogonal(crystal_id, slip_id), 2);

            // Plastic strain tensor
            plastic_strain_tensor +=
              inputs.solution_values[quadrature_point_index](
                  index_offset + slip_id + n_slips * crystal_id) *
              crystals_data->get_symmetrized_schmid_tensor(
                crystal_id, slip_id);
          }
      }
    }

    if (!flag_light_output)
    {
      computed_quantities[quadrature_point_index](dim + n_slips + 2) =
        parameters.characteristic_quantities.dislocation_density *
        std::sqrt(equivalent_edge_dislocation_density);

      computed_quantities[quadrature_point_index](dim + n_slips + 3) =
        parameters.characteristic_quantities.dislocation_density *
        std::sqrt(equivalent_screw_dislocation_density);

      strain_tensor =
        dealii::symmetrize(displacement_gradient) +
          macroscopic_strain;

      elastic_strain_tensor =
        strain_tensor - dimensionless_number * plastic_strain_tensor;

      strain_tensor_3d = convert_2d_to_3d(strain_tensor);

      plastic_strain_tensor_3d =
        convert_2d_to_3d(plastic_strain_tensor);

      elastic_strain_tensor_3d =
        convert_2d_to_3d(elastic_strain_tensor);

      stress_tensor =
        hooke_law->get_stiffness_tetrad_3d(material_id) *
        convert_2d_to_3d(elastic_strain_tensor);

      // Von-Mises stress
      computed_quantities[quadrature_point_index](dim + n_slips + 4) =
        parameters.characteristic_quantities.stress *
        get_von_mises_stress(stress_tensor);

      // Von-Mises plastic strain
      computed_quantities[quadrature_point_index](dim + n_slips + 5) =
        get_von_mises_plastic_strain(plastic_strain_tensor);

      for (unsigned int i = 0; i < voigt_indices.size(); ++i)
      {
        // Stress components
        computed_quantities[quadrature_point_index](dim + n_slips + 6 + i) =
            parameters.characteristic_quantities.stress *
            stress_tensor[voigt_indices[i].first][voigt_indices[i].second];

        // Strain components
        computed_quantities[quadrature_point_index](dim + n_slips + 12 + i) =
          (i < 3 ? 1.0 : 2.0) *
          parameters.characteristic_quantities.strain *
          strain_tensor_3d[voigt_indices[i].first][voigt_indices[i].second];

        // Elastic strain components
        computed_quantities[quadrature_point_index](dim + n_slips + 18 + i) =
          (i < 3 ? 1.0 : 2.0) *
          parameters.characteristic_quantities.strain *
          elastic_strain_tensor_3d[voigt_indices[i].first][voigt_indices[i].second];

        // Plastic strain components
        computed_quantities[quadrature_point_index](dim + n_slips + 24 + i) =
          (i < 3 ? 1.0 : 2.0) *
          plastic_strain_tensor_3d[voigt_indices[i].first][voigt_indices[i].second];
      }

      if (flag_output_fluctuations)
      {
        strain_tensor = dealii::symmetrize(displacement_gradient);

        strain_tensor_3d = convert_2d_to_3d(strain_tensor);

        elastic_strain_tensor =
          strain_tensor - dimensionless_number *
            plastic_strain_tensor;

        stress_tensor =
          parameters.characteristic_quantities.stress *
          hooke_law->get_stiffness_tetrad_3d(material_id) *
          convert_2d_to_3d(elastic_strain_tensor);

        // Von-Mises stress
        computed_quantities[quadrature_point_index](dim + n_slips + 30) =
          parameters.characteristic_quantities.stress *
          get_von_mises_stress(stress_tensor);

        for (unsigned int i = 0; i < voigt_indices.size(); ++i)
        {
          // Stress components
          computed_quantities[quadrature_point_index](dim + n_slips + 31 + i) =
            parameters.characteristic_quantities.stress *
            stress_tensor[voigt_indices[i].first][voigt_indices[i].second];

          // Strain components
          computed_quantities[quadrature_point_index](dim + n_slips + 37 + i) =
            (i < 3 ? 1.0 : 2.0) *
            parameters.characteristic_quantities.strain *
            strain_tensor_3d[voigt_indices[i].first][voigt_indices[i].second];
        }
      }
    }
  }
}



template <int dim>
dealii::SymmetricTensor<2,3>
Postprocessor<dim>::convert_2d_to_3d(
  dealii::SymmetricTensor<2,dim> symmetric_tensor) const
{
  dealii::SymmetricTensor<2,3> symmetric_tensor_in_3d;

  if constexpr(dim == 3)
    symmetric_tensor_in_3d = symmetric_tensor;
  else if constexpr(dim ==2)
  {
    symmetric_tensor_in_3d[0][0] = symmetric_tensor[0][0];
    symmetric_tensor_in_3d[1][1] = symmetric_tensor[1][1];
    symmetric_tensor_in_3d[0][1] = symmetric_tensor[0][1];
  }
  else
    Assert(false, dealii::ExcNotImplemented());

  return (symmetric_tensor_in_3d);
}



template <int dim>
double Postprocessor<dim>::get_von_mises_stress(
  dealii::SymmetricTensor<2,3> stress_tensor_in_3d) const
{
  const dealii::SymmetricTensor<2,3> deviatoric_stress_tensor =
    deviatoric_projector_3d * stress_tensor_in_3d;

  double von_mises_stress =
    deviatoric_stress_tensor * deviatoric_stress_tensor;

  return (std::sqrt(3.0 / 2.0 * von_mises_stress));
}



template <int dim>
double Postprocessor<dim>::get_von_mises_plastic_strain(
  dealii::SymmetricTensor<2,dim> strain_tensor) const
{
  const dealii::SymmetricTensor<2,dim> deviatoric_strain_tensor =
    deviatoric_projector * strain_tensor;

  double von_mises_strain =
    deviatoric_strain_tensor * deviatoric_strain_tensor;

  return (std::sqrt(2.0 / 3.0 * von_mises_strain));
}



template <int dim>
void SlipBasedPostprocessor<dim>::reinit(
  std::shared_ptr<const CrystalsData<dim>> &crystals_data,
  const std::string output_name,
  const unsigned int n_components,
  const bool flag_allow_decohesion)
{
  this->crystals_data = crystals_data;

  this->output_name = output_name;

  this->n_components = n_components;

  this->flag_allow_decohesion = flag_allow_decohesion;
}



template <int dim>
std::vector<std::string>
SlipBasedPostprocessor<dim>::get_names() const
{
  std::vector<std::string> solution_names;

  for (unsigned int slip_id = 0;
      slip_id < crystals_data->get_n_slips();
      ++slip_id)
    solution_names.emplace_back(
      output_name + std::to_string(slip_id));

  return solution_names;
}


template <int dim>
std::vector<dealii::DataComponentInterpretation::DataComponentInterpretation>
SlipBasedPostprocessor<dim>::get_data_component_interpretation()
  const
{
  std::vector<
    dealii::DataComponentInterpretation::DataComponentInterpretation>
      interpretation;

  for (unsigned int slip_id = 0;
      slip_id < crystals_data->get_n_slips();
      ++slip_id)
    interpretation.push_back(
      dealii::DataComponentInterpretation::component_is_scalar);

  return interpretation;
}


template <int dim>
dealii::UpdateFlags
SlipBasedPostprocessor<dim>::get_needed_update_flags() const
{
  return dealii::update_values |
         dealii::update_quadrature_points;
}



template <int dim>
void SlipBasedPostprocessor<dim>::evaluate_vector_field(
  const dealii::DataPostprocessorInputs::Vector<dim>  &inputs,
  std::vector<dealii::Vector<double>>                 &computed_quantities) const
{
  const unsigned int n_q_points   = inputs.solution_values.size();

  const unsigned int n_slips      = crystals_data->get_n_slips();

  const unsigned int n_crystals   = crystals_data->get_n_crystals();

  (void)n_components;

  Assert(inputs.solution_gradients.size() == n_q_points,
         dealii::ExcInternalError());

  Assert(computed_quantities.size() == n_q_points,
         dealii::ExcInternalError());

  Assert(inputs.solution_values[0].size() == n_components,
         dealii::ExcInternalError());

  const unsigned int index_offset =
    flag_allow_decohesion ? dim * n_crystals : dim;


  // Reset
  for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
    for (unsigned int d = 0; d < computed_quantities[0].size(); ++d)
      computed_quantities[q_point](d) = 0.0;

  for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
  {
    for (unsigned int slip_id = 0;
        slip_id < n_slips; ++slip_id)
    {
      for (unsigned int crystal_id = 0;
          crystal_id < n_crystals; ++crystal_id)
      {
        computed_quantities[q_point](slip_id) +=
            std::abs(inputs.solution_values[q_point](
              index_offset + slip_id + n_slips * crystal_id));
      }
    }
  }
}



template <int dim>
SimpleShear<dim>::SimpleShear(
  std::shared_ptr<FEField<dim>>         &fe_field,
  std::shared_ptr<dealii::Mapping<dim>> &mapping,
  const double                          max_shear_strain_at_upper_boundary,
  const double                          min_shear_strain_at_upper_boundary,
  const double                          period,
  const double                          initial_loading_time,
  const RunTimeParameters::LoadingType  loading_type,
  const dealii::types::boundary_id      upper_boundary_id,
  const double                          width)
:
fe_field(fe_field),
mapping_collection(*mapping),
max_shear_strain_at_upper_boundary(max_shear_strain_at_upper_boundary),
min_shear_strain_at_upper_boundary(min_shear_strain_at_upper_boundary),
period(period),
initial_loading_time(initial_loading_time),
loading_type(loading_type),
upper_boundary_id(upper_boundary_id),
width(width),
flag_init_was_called(false)
{
  // Setting up columns
  table_handler.declare_column("shear_at_upper_boundary");
  table_handler.declare_column("stress_12_at_upper_boundary");

  // Setting all columns to scientific notation
  table_handler.set_scientific("shear_at_upper_boundary", true);
  table_handler.set_scientific("stress_12_at_upper_boundary", true);

  // Setting columns' precision
  table_handler.set_precision("shear_at_upper_boundary", 6);
  table_handler.set_precision("stress_12_at_upper_boundary", 6);
}



template <int dim>
void SimpleShear<dim>::init(
  std::shared_ptr<const Kinematics::ElasticStrain<dim>>   elastic_strain,
  std::shared_ptr<const ConstitutiveLaws::HookeLaw<dim>>  hooke_law)
{
  this->elastic_strain  = elastic_strain;
  this->hooke_law       = hooke_law;

  flag_init_was_called = true;
}



template <int dim>
void SimpleShear<dim>::compute_data(const double time)
{
  AssertThrow(flag_init_was_called,
              dealii::ExcMessage("The SimpleShear<dim>"
                                  " instance has not been"
                                  " initialized."));

  compute_stress_12_at_boundary();

  double displacement_load = 0.0;

  switch (loading_type)
  {
    case RunTimeParameters::LoadingType::Monotonic:
      {
        displacement_load =
          time * max_shear_strain_at_upper_boundary;
      }
      break;
    case RunTimeParameters::LoadingType::Cyclic:
      {
        if (time >= initial_loading_time)
          displacement_load =
            (max_shear_strain_at_upper_boundary -
             min_shear_strain_at_upper_boundary) / 2.0 *
            std::cos(2.0 * M_PI / period * (time - initial_loading_time)) +
            (max_shear_strain_at_upper_boundary +
             min_shear_strain_at_upper_boundary) / 2.0;
        else
          displacement_load =
            max_shear_strain_at_upper_boundary *
            std::sin(M_PI / 2.0 / initial_loading_time * time);
      }
      break;
    case RunTimeParameters::LoadingType::CyclicWithUnloading:
      {
        if (time >= initial_loading_time)
          displacement_load =
            (max_shear_strain_at_upper_boundary -
             min_shear_strain_at_upper_boundary) / 2.0 *
            std::cos(2.0 * M_PI / period * (time - initial_loading_time)) +
            (max_shear_strain_at_upper_boundary +
             min_shear_strain_at_upper_boundary) / 2.0;
        else
          displacement_load =
            max_shear_strain_at_upper_boundary *
            std::sin(M_PI / 2.0 / initial_loading_time * time);
      }
      break;
    default:
      Assert(false, dealii::ExcNotImplemented());
  }

  table_handler.add_value("shear_at_upper_boundary", displacement_load);
  table_handler.add_value("stress_12_at_upper_boundary", average_stress_12);
}



template <int dim>
void SimpleShear<dim>::compute_stress_12_at_boundary()
{
  // Initiate the local integral value and at each wall.
  average_stress_12 = 0.0;

  dealii::hp::QCollection<dim-1>  face_quadrature_collection;
  {
    const dealii::QGauss<dim-1>     face_quadrature_formula(3);

    face_quadrature_collection.push_back(face_quadrature_formula);
  }

  const dealii::UpdateFlags face_update_flags =
    dealii::update_JxW_values |
    dealii::update_values |
    dealii::update_gradients;

  // Finite element values
  dealii::hp::FEFaceValues<dim> hp_fe_face_values(
    mapping_collection,
    fe_field->get_fe_collection(),
    face_quadrature_collection,
    face_update_flags);

  // Number of quadrature points
  const unsigned int n_face_q_points =
    face_quadrature_collection.max_n_quadrature_points();

  // Vectors to stores the temperature gradients and normal vectors
  // at the quadrature points
  std::vector<double>                           JxW_values(n_face_q_points);
  std::vector<dealii::SymmetricTensor<2, dim>>  strain_tensor_values(n_face_q_points);
  std::vector<dealii::SymmetricTensor<2, dim>>  elastic_strain_tensor_values(n_face_q_points);
  std::vector<dealii::SymmetricTensor<2, dim>>  stress_tensor_values(n_face_q_points);
  std::vector<std::vector<double>>              slip_values(fe_field->get_n_slips(),
                                                            std::vector<double>(n_face_q_points));

  double stress_12        = 0.0;
  double local_stress_12  = 0.0;

  for (const auto &cell : fe_field->get_dof_handler().active_cell_iterators())
    if (cell->is_locally_owned() && cell->at_boundary())
      for (const auto &face : cell->face_iterators())
        if (face->at_boundary() && face->boundary_id() == upper_boundary_id)
        {
          // Reset local face integral values
          local_stress_12 = 0.0;

          // Get the crystal identifier for the current cell
          const unsigned int crystal_id = cell->material_id();

          // Update the hp::FEFaceValues instance to the values of the current cell
          hp_fe_face_values.reinit(cell, face);

          const dealii::FEFaceValues<dim> &fe_face_values =
            hp_fe_face_values.get_present_fe_values();

          // Get JxW values at the quadrature points
          JxW_values = fe_face_values.get_JxW_values();

          // Get the displacement gradients at the quadrature points
          fe_face_values[fe_field->get_displacement_extractor(crystal_id)].get_function_symmetric_gradients(
            fe_field->solution,
            strain_tensor_values);

          // Get the slips and their gradients values at the quadrature points
          for (unsigned int slip_id = 0;
              slip_id < fe_field->get_n_slips();
              ++slip_id)
          {
            fe_face_values[fe_field->get_slip_extractor(crystal_id, slip_id)].get_function_values(
              fe_field->solution,
              slip_values[slip_id]);
          }

          // Numerical integration
          for (unsigned int face_q_point = 0;
               face_q_point < n_face_q_points;
               ++face_q_point)
          {
            // Compute the elastic strain tensor at the quadrature point
            elastic_strain_tensor_values[face_q_point] =
              elastic_strain->get_elastic_strain_tensor(
                crystal_id,
                face_q_point,
                strain_tensor_values[face_q_point],
                slip_values);

            // Compute the stress tensor at the quadrature point
            stress_tensor_values[face_q_point] =
              hooke_law->get_stress_tensor(
                crystal_id,
                elastic_strain_tensor_values[face_q_point]);

            local_stress_12 +=
              stress_tensor_values[face_q_point][0][1] *
              JxW_values[face_q_point];
          }

          stress_12 += local_stress_12;
        }

  // Gather the values of each processor
  stress_12 = dealii::Utilities::MPI::sum(stress_12, MPI_COMM_WORLD);

  average_stress_12 = stress_12 / width;
}



template <int dim>
void SimpleShear<dim>::output_data_to_file(
  std::ostream &file) const
{
  table_handler.write_text(
    file,
    dealii::TableHandler::TextOutputFormat::org_mode_table);
}



template <int dim>
Homogenization<dim>::Homogenization(
  std::shared_ptr<FEField<dim>>         &fe_field,
  std::shared_ptr<dealii::Mapping<dim>> &mapping)
:
fe_field(fe_field),
mapping_collection(*mapping),
deviatoric_projector(
  dealii::identity_tensor<dim>() -
  1.0 / 3.0 *
  dealii::outer_product(
    dealii::unit_symmetric_tensor<dim>(),
    dealii::unit_symmetric_tensor<dim>())),
flag_init_was_called(false)
{
  const dealii::QGauss<dim> quadrature_formula(3);

  quadrature_collection.push_back(quadrature_formula);

  macroscopic_stress              = 0.;
  microstress_fluctuations        = 0.;
  macroscopic_strain              = 0.;
  microstrain_fluctuations = 0.;

  table_handler.declare_column("Time");
  table_handler.declare_column("VonMisesStress");
  table_handler.declare_column("VonMisesStressFluctuations");
  table_handler.declare_column("VonMisesStrain");
  table_handler.declare_column("VonMisesStrainFluctuations");
  table_handler.declare_column("Stress_11");
  table_handler.declare_column("Stress_22");
  table_handler.declare_column("Stress_33");
  table_handler.declare_column("Stress_23");
  table_handler.declare_column("Stress_13");
  table_handler.declare_column("Stress_12");
  table_handler.declare_column("Strain_11");
  table_handler.declare_column("Strain_22");
  table_handler.declare_column("Strain_33");
  table_handler.declare_column("Strain_23");
  table_handler.declare_column("Strain_13");
  table_handler.declare_column("Strain_12");
  table_handler.declare_column("StressFluctuations_11");
  table_handler.declare_column("StressFluctuations_22");
  table_handler.declare_column("StressFluctuations_33");
  table_handler.declare_column("StressFluctuations_23");
  table_handler.declare_column("StressFluctuations_13");
  table_handler.declare_column("StressFluctuations_12");
  table_handler.declare_column("StrainFluctuations_11");
  table_handler.declare_column("StrainFluctuations_22");
  table_handler.declare_column("StrainFluctuations_33");
  table_handler.declare_column("StrainFluctuations_23");
  table_handler.declare_column("StrainFluctuations_13");
  table_handler.declare_column("StrainFluctuations_12");
  table_handler.set_scientific("VonMisesStress", true);
  table_handler.set_scientific("VonMisesStressFluctuations", true);
  table_handler.set_scientific("VonMisesStrain", true);
  table_handler.set_scientific("VonMisesStrainFluctuations", true);
  table_handler.set_scientific("Stress_11", true);
  table_handler.set_scientific("Stress_22", true);
  table_handler.set_scientific("Stress_33", true);
  table_handler.set_scientific("Stress_23", true);
  table_handler.set_scientific("Stress_13", true);
  table_handler.set_scientific("Stress_12", true);
  table_handler.set_scientific("Strain_11", true);
  table_handler.set_scientific("Strain_22", true);
  table_handler.set_scientific("Strain_33", true);
  table_handler.set_scientific("Strain_23", true);
  table_handler.set_scientific("Strain_13", true);
  table_handler.set_scientific("Strain_12", true);
  table_handler.set_scientific("StressFluctuations_11", true);
  table_handler.set_scientific("StressFluctuations_22", true);
  table_handler.set_scientific("StressFluctuations_33", true);
  table_handler.set_scientific("StressFluctuations_23", true);
  table_handler.set_scientific("StressFluctuations_13", true);
  table_handler.set_scientific("StressFluctuations_12", true);
  table_handler.set_scientific("StrainFluctuations_11", true);
  table_handler.set_scientific("StrainFluctuations_22", true);
  table_handler.set_scientific("StrainFluctuations_33", true);
  table_handler.set_scientific("StrainFluctuations_23", true);
  table_handler.set_scientific("StrainFluctuations_13", true);
  table_handler.set_scientific("StrainFluctuations_12", true);
  table_handler.set_precision("VonMisesStress", 18);
  table_handler.set_precision("VonMisesStressFluctuations", 18);
  table_handler.set_precision("VonMisesStrain", 18);
  table_handler.set_precision("VonMisesStrainFluctuations", 18);
  table_handler.set_precision("Stress_11", 18);
  table_handler.set_precision("Stress_22", 18);
  table_handler.set_precision("Stress_33", 18);
  table_handler.set_precision("Stress_23", 18);
  table_handler.set_precision("Stress_13", 18);
  table_handler.set_precision("Stress_12", 18);
  table_handler.set_precision("Strain_11", 18);
  table_handler.set_precision("Strain_22", 18);
  table_handler.set_precision("Strain_33", 18);
  table_handler.set_precision("Strain_23", 18);
  table_handler.set_precision("Strain_13", 18);
  table_handler.set_precision("Strain_12", 18);
  table_handler.set_precision("StressFluctuations_11", 18);
  table_handler.set_precision("StressFluctuations_22", 18);
  table_handler.set_precision("StressFluctuations_33", 18);
  table_handler.set_precision("StressFluctuations_23", 18);
  table_handler.set_precision("StressFluctuations_13", 18);
  table_handler.set_precision("StressFluctuations_12", 18);
  table_handler.set_precision("StrainFluctuations_11", 18);
  table_handler.set_precision("StrainFluctuations_22", 18);
  table_handler.set_precision("StrainFluctuations_33", 18);
  table_handler.set_precision("StrainFluctuations_23", 18);
  table_handler.set_precision("StrainFluctuations_13", 18);
  table_handler.set_precision("StrainFluctuations_12", 18);
}



template <int dim>
void Homogenization<dim>::init(
  std::shared_ptr<const Kinematics::ElasticStrain<dim>>   elastic_strain,
  std::shared_ptr<const ConstitutiveLaws::HookeLaw<dim>>  hooke_law,
  std::ofstream                                           &path_to_output_file)
{
  this->elastic_strain  = elastic_strain;
  this->hooke_law       = hooke_law;
  this->path_to_output_file.swap(path_to_output_file);

  flag_init_was_called = true;
}



template <int dim>
void Homogenization<dim>::compute_macroscopic_quantities(const double time)
{
  AssertThrow(flag_init_was_called,
              dealii::ExcMessage("The HookeLaw<dim> instance has not"
                                 " been initialized."));

  compute_macroscopic_stress();

  compute_macroscopic_stiffness_tetrad();

  update_table_handler_values(time);
}



template <int dim>
void Homogenization<dim>::output_macroscopic_quantities_to_file()
{
  table_handler.write_text(
    path_to_output_file,
    dealii::TableHandler::TextOutputFormat::org_mode_table);
}



template <>
void Homogenization<2>::update_table_handler_values(const double time)
{
  const dealii::SymmetricTensor<2,2> deviatoric_stress =
    deviatoric_projector * macroscopic_stress;

  const dealii::SymmetricTensor<2,2> deviatoric_stress_fluctuations =
    deviatoric_projector * microstress_fluctuations;

  const dealii::SymmetricTensor<2,2> deviatoric_strain =
    deviatoric_projector * (macroscopic_strain +
                            microstrain_fluctuations);

  const dealii::SymmetricTensor<2,2> deviatoric_strain_fluctuations =
    deviatoric_projector * microstrain_fluctuations;

  const double von_mises_stress =
    std::sqrt(3./2. * deviatoric_stress * deviatoric_stress);

  const double von_mises_stress_fluctuations =
    std::sqrt(3./2. * deviatoric_stress_fluctuations *
              deviatoric_stress_fluctuations);

  const double von_mises_strain =
    std::sqrt(2./3. * deviatoric_strain * deviatoric_strain);

  const double von_mises_strain_fluctuations =
    std::sqrt(2./3. * deviatoric_strain_fluctuations *
              deviatoric_strain_fluctuations);

  table_handler.add_value("Time", time);
  table_handler.add_value("VonMisesStress", von_mises_stress);
  table_handler.add_value("VonMisesStressFluctuations", von_mises_stress_fluctuations);
  table_handler.add_value("VonMisesStrain", von_mises_strain);
  table_handler.add_value("VonMisesStrainFluctuations", von_mises_strain_fluctuations);
  table_handler.add_value("Stress_11", macroscopic_stress[0][0]);
  table_handler.add_value("Stress_22", macroscopic_stress[1][1]);
  table_handler.add_value("Stress_12", macroscopic_stress[0][1]);
  table_handler.add_value("Strain_11", macroscopic_strain[0][0] + microstrain_fluctuations[0][0]);
  table_handler.add_value("Strain_22", macroscopic_strain[1][1] + microstrain_fluctuations[1][1]);
  table_handler.add_value("Strain_12", macroscopic_strain[0][1] + microstrain_fluctuations[0][1]);
  table_handler.add_value("StressFluctuations_11", microstress_fluctuations[0][0]);
  table_handler.add_value("StressFluctuations_22", microstress_fluctuations[1][1]);
  table_handler.add_value("StressFluctuations_12", microstress_fluctuations[0][1]);
  table_handler.add_value("StrainFluctuations_11", microstrain_fluctuations[0][0]);
  table_handler.add_value("StrainFluctuations_22", microstrain_fluctuations[1][1]);
  table_handler.add_value("StrainFluctuations_12", microstrain_fluctuations[0][1]);

  table_handler.start_new_row();
}



template <>
void Homogenization<3>::update_table_handler_values(const double time)
{
  const dealii::SymmetricTensor<2,3> deviatoric_stress =
    deviatoric_projector * macroscopic_stress;

  const dealii::SymmetricTensor<2,3> deviatoric_stress_fluctuations =
    deviatoric_projector * microstress_fluctuations;

  const dealii::SymmetricTensor<2,3> deviatoric_strain =
    deviatoric_projector * (macroscopic_strain +
                            microstrain_fluctuations);

  const dealii::SymmetricTensor<2,3> deviatoric_strain_fluctuations =
    deviatoric_projector * microstrain_fluctuations;

  const double von_mises_stress =
    std::sqrt(3./2. * deviatoric_stress * deviatoric_stress);

  const double von_mises_stress_fluctuations =
    std::sqrt(3./2. * deviatoric_stress_fluctuations *
              deviatoric_stress_fluctuations);

  const double von_mises_strain =
    std::sqrt(2./3. * deviatoric_strain * deviatoric_strain);

  const double von_mises_strain_fluctuations =
    std::sqrt(2./3. * deviatoric_strain_fluctuations *
              deviatoric_strain_fluctuations);

  table_handler.add_value("Time", time);
  table_handler.add_value("VonMisesStress", von_mises_stress);
  table_handler.add_value("VonMisesStressFluctuations", von_mises_stress_fluctuations);
  table_handler.add_value("VonMisesStrain", von_mises_strain);
  table_handler.add_value("VonMisesStrainFluctuations", von_mises_strain_fluctuations);
  table_handler.add_value("Stress_11", macroscopic_stress[0][0]);
  table_handler.add_value("Stress_22", macroscopic_stress[1][1]);
  table_handler.add_value("Stress_33", macroscopic_stress[2][2]);
  table_handler.add_value("Stress_23", macroscopic_stress[1][2]);
  table_handler.add_value("Stress_13", macroscopic_stress[0][2]);
  table_handler.add_value("Stress_12", macroscopic_stress[0][1]);
  table_handler.add_value("Strain_11", macroscopic_strain[0][0] + microstrain_fluctuations[0][0]);
  table_handler.add_value("Strain_22", macroscopic_strain[1][1] + microstrain_fluctuations[1][1]);
  table_handler.add_value("Strain_33", macroscopic_strain[2][2] + microstrain_fluctuations[2][2]);
  table_handler.add_value("Strain_23", macroscopic_strain[1][2] + microstrain_fluctuations[1][2]);
  table_handler.add_value("Strain_13", macroscopic_strain[0][2] + microstrain_fluctuations[0][2]);
  table_handler.add_value("Strain_12", macroscopic_strain[0][1] + microstrain_fluctuations[0][1]);
  table_handler.add_value("StressFluctuations_11", microstress_fluctuations[0][0]);
  table_handler.add_value("StressFluctuations_22", microstress_fluctuations[1][1]);
  table_handler.add_value("StressFluctuations_33", microstress_fluctuations[2][2]);
  table_handler.add_value("StressFluctuations_23", microstress_fluctuations[1][2]);
  table_handler.add_value("StressFluctuations_13", microstress_fluctuations[0][2]);
  table_handler.add_value("StressFluctuations_12", microstress_fluctuations[0][1]);
  table_handler.add_value("StrainFluctuations_11", microstrain_fluctuations[0][0]);
  table_handler.add_value("StrainFluctuations_22", microstrain_fluctuations[1][1]);
  table_handler.add_value("StrainFluctuations_33", microstrain_fluctuations[2][2]);
  table_handler.add_value("StrainFluctuations_23", microstrain_fluctuations[1][2]);
  table_handler.add_value("StrainFluctuations_13", microstrain_fluctuations[0][2]);
  table_handler.add_value("StrainFluctuations_12", microstrain_fluctuations[0][1]);

  table_handler.start_new_row();
}



template <int dim>
void Homogenization<dim>::set_macroscopic_strain(
  const dealii::SymmetricTensor<2,dim> macroscopic_strain)
{
  for (unsigned int i = 0;
        i < macroscopic_strain.n_independent_components;
        ++i)
    AssertIsFinite(macroscopic_strain.access_raw_entry(i));

  this->macroscopic_strain = macroscopic_strain;
}



template <int dim>
void Homogenization<dim>::compute_macroscopic_stress()
{
  // Initiate the local integral value and at each wall.
  macroscopic_stress              = 0.0;

  microstress_fluctuations = 0.0;

  microstrain_fluctuations = 0.0;

  const dealii::UpdateFlags update_flags =
    dealii::update_JxW_values |
    dealii::update_values |
    dealii::update_gradients;

  // Finite element values
  dealii::hp::FEValues<dim> hp_fe_values(
    mapping_collection,
    fe_field->get_fe_collection(),
    quadrature_collection,
    update_flags);

  // Number of quadrature points
  const unsigned int n_quadrature_points =
    quadrature_collection.max_n_quadrature_points();

  // Vectors to stores the temperature gradients and normal vectors
  // at the quadrature points
  std::vector<double>                           JxW_values(n_quadrature_points);
  std::vector<dealii::SymmetricTensor<2, dim>>  strain_tensor_values(n_quadrature_points);
  std::vector<dealii::SymmetricTensor<2, dim>>  elastic_strain_tensor_values(n_quadrature_points);
  std::vector<dealii::SymmetricTensor<2, dim>>  stress_tensor_values(n_quadrature_points);
  std::vector<dealii::SymmetricTensor<2, dim>>  stress_tensor_fluctuations_values(n_quadrature_points);
  std::vector<std::vector<double>>              slip_values(fe_field->get_n_slips(),
                                                            std::vector<double>(n_quadrature_points));

  dealii::SymmetricTensor<2,dim>  domain_integral_microstress;

  dealii::SymmetricTensor<2,dim>  domain_integral_microstress_fluctuations;

  dealii::SymmetricTensor<2,dim>  domain_integral_microstrain_fluctuations;

  dealii::SymmetricTensor<2,dim>  cell_integral_microstress;

  dealii::SymmetricTensor<2,dim>  cell_integral_microstress_fluctuations;

  dealii::SymmetricTensor<2,dim>  cell_integral_microstrain_fluctuations;

  double                          domain_volume = 0.;

  double                          cell_volume = 0.;

  domain_integral_microstress              = 0.;

  domain_integral_microstress_fluctuations = 0.;

  domain_integral_microstrain_fluctuations = 0.;

  for (const auto &cell : fe_field->get_dof_handler().active_cell_iterators())
  {
    if (cell->is_locally_owned())
    {
      // Reset local values
      cell_integral_microstress              = 0.0;

      cell_integral_microstress_fluctuations = 0.0;

      cell_integral_microstrain_fluctuations = 0.0;

      cell_volume                                   = 0.0;

      // Get the crystal identifier for the current cell
      const unsigned int crystal_id = cell->material_id();

      // Update the hp::FEFaceValues instance to the values of the current cell
      hp_fe_values.reinit(cell);

      const dealii::FEValues<dim> &fe_values =
        hp_fe_values.get_present_fe_values();

      // Get JxW values at the quadrature points
      JxW_values = fe_values.get_JxW_values();

      // Get the displacement gradients at the quadrature points
      fe_values[fe_field->get_displacement_extractor(crystal_id)].get_function_symmetric_gradients(
        fe_field->solution,
        strain_tensor_values);

      // Get the slips values at the quadrature points
      for (unsigned int slip_id = 0;
          slip_id < fe_field->get_n_slips();
          ++slip_id)
      {
        fe_values[fe_field->get_slip_extractor(crystal_id, slip_id)].get_function_values(
          fe_field->solution,
          slip_values[slip_id]);
      }

      // Numerical integration
      for (unsigned int quadrature_point_id = 0;
            quadrature_point_id < n_quadrature_points;
            ++quadrature_point_id)
      {
        // Compute the elastic strain tensor at the quadrature point
        elastic_strain_tensor_values[quadrature_point_id] =
          elastic_strain->get_elastic_strain_tensor(
            crystal_id,
            quadrature_point_id,
            strain_tensor_values[quadrature_point_id],
            slip_values);

        // Compute the stress tensor at the quadrature point
        stress_tensor_values[quadrature_point_id] =
          hooke_law->get_stress_tensor(
            crystal_id,
            macroscopic_strain +
            elastic_strain_tensor_values[quadrature_point_id]);

        stress_tensor_fluctuations_values[quadrature_point_id] =
          hooke_law->get_stress_tensor(
            crystal_id,
            elastic_strain_tensor_values[quadrature_point_id]);

        cell_integral_microstress +=
          stress_tensor_values[quadrature_point_id] *
          JxW_values[quadrature_point_id];

        cell_integral_microstress_fluctuations +=
          stress_tensor_fluctuations_values[quadrature_point_id] *
          JxW_values[quadrature_point_id];

        cell_integral_microstrain_fluctuations +=
          strain_tensor_values[quadrature_point_id] *
          JxW_values[quadrature_point_id];

        cell_volume += JxW_values[quadrature_point_id];
      }

      domain_integral_microstress +=
        cell_integral_microstress;

      domain_integral_microstress_fluctuations +=
        cell_integral_microstress_fluctuations;

      domain_integral_microstrain_fluctuations +=
        cell_integral_microstrain_fluctuations;

      domain_volume += cell_volume;
    }
  }

  // Gather the values of each processor
  domain_integral_microstress =
    dealii::Utilities::MPI::sum(domain_integral_microstress,
                                MPI_COMM_WORLD);

  domain_integral_microstress_fluctuations =
    dealii::Utilities::MPI::sum(
      domain_integral_microstress_fluctuations,
      MPI_COMM_WORLD);

  domain_integral_microstrain_fluctuations =
    dealii::Utilities::MPI::sum(
      domain_integral_microstrain_fluctuations, MPI_COMM_WORLD);

  domain_volume =
    dealii::Utilities::MPI::sum(domain_volume, MPI_COMM_WORLD);

  // Compute the homogenized values
  macroscopic_stress = domain_integral_microstress / domain_volume;

  microstress_fluctuations =
    domain_integral_microstress_fluctuations / domain_volume;

  microstrain_fluctuations =
    domain_integral_microstrain_fluctuations / domain_volume;
}



template <int dim>
void Homogenization<dim>::compute_macroscopic_stiffness_tetrad()
{

}



} // namespace Postprocessing



} // namespace gCP



template class gCP::Postprocessing::Postprocessor<2>;
template class gCP::Postprocessing::Postprocessor<3>;

template class gCP::Postprocessing::SlipBasedPostprocessor<2>;
template class gCP::Postprocessing::SlipBasedPostprocessor<3>;

template class gCP::Postprocessing::SimpleShear<2>;
template class gCP::Postprocessing::SimpleShear<3>;

template class gCP::Postprocessing::Homogenization<2>;
template class gCP::Postprocessing::Homogenization<3>;
