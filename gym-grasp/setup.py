from setuptools import setup

setup(name='gym_grasp',
      version='0.0.1',
      install_requires=['gym>=0.2.3',
                        'mujoco_py>=1.50'],
      package_data={'gym_grasp' : [
          'envs/assets/hand/*.xml'
      ]}
)
