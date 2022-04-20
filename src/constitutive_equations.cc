#include <gCP/constitutive_equations.h>



namespace gCP
{



namespace ConstitutiveEquations
{



template<int dim>
HookeLaw<dim>::HookeLaw(const double youngs_modulus,
                        const double poissons_ratio)
:
bulk_modulus(youngs_modulus/(3.0 * (1.0 - 2.0 * poissons_ratio))),
shear_modulus(youngs_modulus/(2.0 * (1.0 + poissons_ratio))),
spherical_projector(1.0 / 3.0 * dealii::outer_product(
                                  dealii::unit_symmetric_tensor<dim>(),
                                  dealii::unit_symmetric_tensor<dim>())),
deviatoric_projector(dealii::identity_tensor<dim>() - spherical_projector)
{
  compute_stiffness_tetrad();
}



template<int dim>
void HookeLaw<dim>::compute_stiffness_tetrad()
{
  stiffness_tetrad = 3.0 * bulk_modulus * spherical_projector
                      + 2.0 * shear_modulus * deviatoric_projector;
}



template<int dim>
const dealii::SymmetricTensor<2,dim> HookeLaw<dim>::
compute_stress_tensor(const dealii::SymmetricTensor<2,dim> strain_tensor_values) const
{
  return stiffness_tetrad * strain_tensor_values;
}



} // ConstitutiveEquations



} // gCP



template class gCP::ConstitutiveEquations::HookeLaw<2>;
template class gCP::ConstitutiveEquations::HookeLaw<3>;
