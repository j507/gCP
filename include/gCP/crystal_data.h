#ifndef INCLUDE_CRYSTAL_DATA_H_
#define INCLUDE_CRYSTAL_DATA_H_

#include <deal.II/base/tensor.h>

#include <deal.II/base/conditional_ostream.h>

#include <deal.II/distributed/tria.h>


namespace gCP
{



template<int dim>
class CrystalsData
{
public:

  CrystalsData();

  void init(const dealii::Triangulation<dim>  &triangulation,
            const std::string                 crystal_orientations_file_name,
            const std::string                 slip_directions_file_name,
            const std::string                 slip_normals_file_name);

  const unsigned int &get_n_crystals() const;

  const unsigned int &get_n_slips() const;

  const dealii::Tensor<2,dim> &get_rotation_matrix(
    const unsigned int crystal_id) const;

  const dealii::Tensor<1,dim> &get_slip_direction(
    const unsigned int crystal_id,
    const unsigned int slip_id) const;

  const dealii::Tensor<1,dim> &get_slip_normal(
    const unsigned int crystal_id,
    const unsigned int slip_id) const;

  const dealii::Tensor<1,dim> &get_slip_orthogonal(
    const unsigned int crystal_id,
    const unsigned int slip_id) const;

  const dealii::Tensor<2,dim> &get_schmid_tensor(
    const unsigned int crystal_id,
    const unsigned int slip_id) const;

  const dealii::Tensor<2,dim> &get_symmetrized_schmid_tensor(
    const unsigned int crystal_id,
    const unsigned int slip_id) const;

private:
  dealii::ConditionalOStream                      pcout;

  unsigned int                                    n_crystals;

  unsigned int                                    n_slips;

  std::vector<dealii::Tensor<1,dim>>              euler_angles;

  std::vector<dealii::Tensor<2,dim>>              rotation_matrices;

  std::vector<dealii::Tensor<1,dim>>              reference_slip_directions;

  std::vector<dealii::Tensor<1,dim>>              reference_slip_normals;

  std::vector<dealii::Tensor<1,dim>>              reference_slip_orthogonals;

  std::vector<std::vector<dealii::Tensor<1,dim>>> slip_directions;

  std::vector<std::vector<dealii::Tensor<1,dim>>> slip_normals;

  std::vector<std::vector<dealii::Tensor<1,dim>>> slip_orthogonals;

  std::vector<std::vector<dealii::Tensor<2,dim>>> schmid_tensors;

  std::vector<std::vector<dealii::Tensor<2,dim>>> symmetrized_schmid_tensors;

  bool                                            flag_init_was_called;

  void count_n_crystals(const dealii::Triangulation<dim>  &triangulation);

  void read_and_store_data(
    const std::string crystal_orientations_file_name,
    const std::string slip_directions_file_name,
    const std::string slip_normals_file_name);

  void compute_rotation_matrices();

  void compute_slip_systems();

  bool is_initialized();

  bool orthogonality_check(const dealii::Tensor<1,dim> a,
                           const dealii::Tensor<1,dim> b);

};



template <int dim>
inline const unsigned int
&CrystalsData<dim>::get_n_crystals() const
{
  return (n_crystals);
}



template <int dim>
inline const unsigned int
&CrystalsData<dim>::get_n_slips() const
{
  return (n_slips);
}



template <int dim>
inline const dealii::Tensor<2,dim>
&CrystalsData<dim>::get_rotation_matrix(const unsigned int crystal_id) const
{
  dealii::ExcIndexRangeType<int>(crystal_id,0,n_crystals);
  return (rotation_matrices[crystal_id]);
}



template <int dim>
inline const dealii::Tensor<1,dim>
&CrystalsData<dim>::get_slip_direction(const unsigned int crystal_id,
                                       const unsigned int slip_id) const
{
  dealii::ExcIndexRangeType<int>(crystal_id,0,n_crystals);
  dealii::ExcIndexRangeType<int>(slip_id,0,n_slips);
  return (slip_directions[crystal_id][slip_id]);
}



template <int dim>
inline const dealii::Tensor<1,dim>
&CrystalsData<dim>::get_slip_normal(const unsigned int crystal_id,
                                    const unsigned int slip_id) const
{
  dealii::ExcIndexRangeType<int>(crystal_id,0,n_crystals);
  dealii::ExcIndexRangeType<int>(slip_id,0,n_slips);
  return (slip_normals[crystal_id][slip_id]);
}



template <int dim>
inline const dealii::Tensor<1,dim>
&CrystalsData<dim>::get_slip_orthogonal(const unsigned int crystal_id,
                                        const unsigned int slip_id) const
{
  dealii::ExcIndexRangeType<int>(crystal_id,0,n_crystals);
  dealii::ExcIndexRangeType<int>(slip_id,0,n_slips);
  return (slip_orthogonals[crystal_id][slip_id]);
}



template <int dim>
inline const dealii::Tensor<2,dim>
&CrystalsData<dim>::get_schmid_tensor(const unsigned int crystal_id,
                                      const unsigned int slip_id) const
{
  dealii::ExcIndexRangeType<int>(crystal_id,0,n_crystals);
  dealii::ExcIndexRangeType<int>(slip_id,0,n_slips);
  return (schmid_tensors[crystal_id][slip_id]);
}



template <int dim>
inline const dealii::Tensor<2,dim>
&CrystalsData<dim>::get_symmetrized_schmid_tensor(
  const unsigned int crystal_id,
  const unsigned int slip_id) const
{
  dealii::ExcIndexRangeType<int>(crystal_id,0,n_crystals);
  dealii::ExcIndexRangeType<int>(slip_id,0,n_slips);
  return (symmetrized_schmid_tensors[crystal_id][slip_id]);
}



}  // namespace gCP



#endif /* INCLUDE_CRYSTAL_DATA_H_ */
