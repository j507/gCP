#ifndef INCLUDE_CONSTITUTIVE_EQUATIONS_H_
#define INCLUDE_CONSTITUTIVE_EQUATIONS_H_

#include <gCP/crystal_data.h>

#include <deal.II/base/symmetric_tensor.h>

#include <deal.II/fe/fe_values.h>

#include <deal.II/lac/generic_linear_algebra.h>



namespace gCP
{



namespace Kinematics
{


template <int dim>
class ElasticStrainTensor
{
public:
  ElasticStrainTensor();

  const std::vector<dealii::SymmetricTensor<2,dim>> get_elastic_strain_tensor(
    const dealii::LinearAlgebraTrilinos::MPI::Vector  solution,
    const dealii::FEValues<dim>                       &fe_values,
    const dealii::types::material_id                  crystal_id) const;

private:

  std::shared_ptr<CrystalsData<dim>>              crystals_data;

  std::shared_ptr<std::vector<int>>               displacements_extractors;

  std::shared_ptr<std::vector<std::vector<int>>>  slips_extractors;
};

template <int dim>
const std::vector<dealii::SymmetricTensor<2,dim>>
ElasticStrainTensor<dim>::get_elastic_strain_tensor(
  const dealii::LinearAlgebraTrilinos::MPI::Vector  solution,
  const dealii::FEValues<dim>                       &fe_values,
  const dealii::types::material_id                  crystal_id) const
{
  const unsigned int n_q_points = fe_values.n_quadrature_points;

  std::vector<dealii::SymmetricTensor<2,dim>> strain_tensor_values(
    n_q_points,
    dealii::SymmetricTensor<2,dim>());

  std::vector<dealii::SymmetricTensor<2,dim>> elastic_strain_tensor_values(
    n_q_points,
    dealii::SymmetricTensor<2,dim>());

  std::vector<dealii::SymmetricTensor<2,dim>> plastic_strain_tensor_values(
    n_q_points,
    dealii::SymmetricTensor<2,dim>());

  std::vector<double>                         slip_values(n_q_points, 0);

  dealii::SymmetricTensor<2,dim>              schmid_tensor();

  fe_values[(*displacements_extractors)[crystal_id]]->get_function_symmetric_gradients(
    solution,
    strain_tensor_values);

  elastic_strain_tensor_values  = strain_tensor_values;

  for (unsigned int slip_id = 0; slip_id < (*crystals_data)[crystal_id].n_slips; ++slip_id)
  {
    fe_values[(*slips_extractors)[crystal_id][slip_id]]->get_function_value(
      solution,
      slip_values);

    schmid_tensor =
      crystals_data.get_symmetrized_schmid_tensor(crystal_id, slip_id);

    for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
    {
      elastic_strain_tensor_values[q_point] -=
        slip_values[q_point] * schmid_tensor;
    }
  }

  return elastic_strain_tensor_values;
}



} // namespace Kinematics


namespace ConstitutiveEquations
{



template<int dim>
class HookeLaw
{
public:
  HookeLaw(const double youngs_modulus,
           const double poissons_ratio);

  HookeLaw(const std::shared_ptr<CrystalsData<dim>> &crystals_data,
           const double                             C_1111,
           const double                             C_1212,
           const double                             C_1122);

  void init();

  void compute_reference_stiffness_tetrad();

  const dealii::SymmetricTensor<4,dim> &get_stiffness_tetrad() const;

  const dealii::SymmetricTensor<4,dim> &get_stiffness_tetrad(
    const unsigned int crystal_id) const;

  const dealii::SymmetricTensor<2,dim> compute_stress_tensor(
    const dealii::SymmetricTensor<2,dim> strain_tensor_values) const;

  const dealii::SymmetricTensor<2,dim> compute_stress_tensor(
    const unsigned int                    crystal_id,
    const dealii::SymmetricTensor<2,dim>  strain_tensor_values) const;

private:
  enum class CrystalSystem
  {
    Isotropic,
    Cubic
  };

  std::shared_ptr<CrystalsData<dim>>          crystals_data;

  CrystalSystem                               crystal_system;

  const double                                C_1111;

  const double                                C_1212;

  const double                                C_1122;

  dealii::SymmetricTensor<4,dim>              reference_stiffness_tetrad;

  std::vector<dealii::SymmetricTensor<4,dim>> stiffness_tetrads;

  bool                                        flag_init_was_called;
};



template <int dim>
inline const dealii::SymmetricTensor<4,dim>
&HookeLaw<dim>::get_stiffness_tetrad() const
{
  return (reference_stiffness_tetrad);
}



template <int dim>
inline const dealii::SymmetricTensor<4,dim>
&HookeLaw<dim>::get_stiffness_tetrad(const unsigned int crystal_id) const
{
  return (stiffness_tetrads[crystal_id]);
}


template<int dim, int n_slips>
class ResolvedShearStressLaw
{

};




template<int dim, int n_slips>
class ScalarMicroscopicStressLaw
{

};



template<int dim, int n_slips>
class VectorMicroscopicStressLaw
{

};



template<int dim, int n_slips>
class MicroscopicTractionLaw
{

};



template<int dim, int n_slips>
class MicroscopicInterfaceTractionLaw
{

};



} // ConstitutiveEquations



} // gCP



#endif /* INCLUDE_CONSTITUTIVE_EQUATIONS_H_ */