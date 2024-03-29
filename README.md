<div align="center">
  <img src="https://github.com/parham/parham.github.io/blob/main/assets/img/favicon.png"/ width="200">
</div>

<!-- PROJECT LOGO -->
<br />
<p align="center">
  <a href="https://www.ulaval.ca/en/" target="_blank">
    <img src="https://ssc.ca/sites/default/files/logo-ulaval-reseaux-sociaux.jpg" alt="Logo" width="200" height="100">
  </a>

  <h3 align="center">LeManchot-Points</h3>

  <p align="center">
	The fusion subsystem of the LeManchot platform. It provides the required tools for fusion and registration of RGBD&T data.
    <br/>
    <br/>
  </p>
</p>

## LeManchot-Points

Thermography is a Non-Destructive Testing (NDT) technology that measures the thermal distribution of a specimen by quantifying the electromagnetic radiation emitted, reflected and transmitted at lower frequency than the visible light part of the spectrum. Despite some previous studies addressing the estimation of surface and shape characterization from multiple or single active thermograms, thermography, by nature, is a bi-dimensional sensing technology unable to provide information about the specimen's texture and geometry without any preparatory process. Thus, many studies have recently focused on using multi-modal platforms to obtain extensive information about the inspected scene. Still, data availability and algorithm implementation difficulties are faced by the analysts when performing the registration of consecutive 3D data from multiple sensors and Fields of Views (FOVs). This study presents a complete solution for multi-modal inspection of industrial components, including a processing pipeline for registering consecutive multi-modal point clouds. A comparative evaluation of optimization and learning based registration methods is provided as part of the processing pipeline. Moreover, a benchmark dataset of point cloud data from different FOVs of industrial and construction components is provided (Lemanchot-points), having 5 point clouds with depth, color, and thermal information at each point. The  experimental campaign conducted with different objects demonstrates the proposed solution's applicability for the multi-modal inspection of industrial components. 

<p align="center">
  <img src="resources/mme_design.png" width="800" title="Abstract View">
</p>

## Citation

```
@article{nooralishahi2023registration,
  title={The registration of multi-modal point clouds for industrial inspection},
  author={Nooralishahi, Parham and Pozzer, Sandra and Ramos, Gabriel and L{\'o}pez, Fernando and Maldague, Xavier},
  journal={Quantitative InfraRed Thermography Journal},
  pages={1--18},
  year={2023},
  publisher={Taylor \& Francis}
}
```


## Dataset 

Dataset provided in this study, is available in Mendeley Data

https://data.mendeley.com/datasets/2mt5sxp75j/1

Nooralishahi, Parham; Pozzer, Sandra; Ramos, Gabriel (2022), “Multi-Modal Inspection of Industrial Structures v1.0”, Mendeley Data, V1, doi: 10.17632/2mt5sxp75j.1


### Multi-Modal Visualizer

<p align="center">
  <img src="resources/pc_fusion.gif" width="800" title="Visualization of piping inspection">
</p>

<p align="center">
  <img src="resources/cli_tools.gif" width="800" title="Visualization of piping inspection">
</p>

<p align="center">
  <img src="resources/concrete_vertical.gif" width="800" title="Visualization of concrete structure">
</p>

### Noise Reduction

<p align="center">
  <img src="resources/noise_reduction.png" width="800" title="Results of Noise Reduction">
</p>

### Results

<p align="center">
  <img src="resources/metrics.png" width="800" title="Metrics">
</p>

<p align="center">
  <img src="resources/results.png" width="800" title="Sample Results">
</p>

<p align="center">
  <img src="resources/iteration_results.png" width="800" title="Result of Iteration Analysis">
</p>

## Contributors
**Parham Nooralishahi** - parham.nooralishahi@gmail.com | [@phm](https://www.linkedin.com/in/parham-nooralishahi/) <br/>
**Sandra Pozzer** - sandra.pozzer.1@ulaval.ca | [@sandra](https://www.linkedin.com/in/sandra-pozzer/) <br/>
**Gabriel Ramos** - gabriel.ramos.1@ulaval.ca | [@gabriel](https://www.linkedin.com/in/gramos-ing/) <br/>

## Team
**Parham Nooralishahi** is a specialist in embedded and intelligent vision systems and currently is a Ph.D. student at Universite Laval working on developing drone-enabled techniques for the inspection of large and complex industrial components using multi-modal data processing. He is a researcher with a demonstrated history of working in the telecommunication industry and industrial inspection and in-depth expertise in robotics & drones, embedded systems, advanced computer vision and machine learning techniques. He has a Master’s degree in Computer Science (Artificial Intelligence). During his bachelor's degree, he was involved in designing and developing the controlling and monitoring systems for fixed-wing drone for search and rescue purposes. Also, during his Master's degree, he worked extensively on machine learning and computer vision techniques for robotic and soft computing applications.

**Sandra Pozzer** received his Bachelor's degree in Civil Engineering (B.Eng.) from the University of Passo Fundo, Brazil (2016). During his bachelor's degree, she specialized in transportation Infrastructure, including one year of applied studies at the Università Degli Studi di Padova, Italy (2013-2014). During her Master's studies, she studied infrared thermography applied to the inspection of concrete structures at the University of Passo Fundo, Brazil ( 2020), including one term of applied research at Lakehead University, Ontario, Canada in 2019. Currently, she is a Ph.D. candidate at Laval Université, Quebec, Canadá, exploring the subjects of infrared thermography, concrete infrastructure, and computer vision applied to civil infrastructure. She has professional experience in the fields of concrete structures, topography, transportation, and infrastructure projects.

**Gabriel Ramos** received his Bachelor's degree in Mechanical Engineering (B.Eng.) from Universit\'e Laval, Quebec, Canada in 2017. 
During his bachelor's degree and his early career, he specialised in numerical structural, modal, and thermal simulations, data analysis and mechanical systems design. He is currently a student in the department of Computer Science and Software Engineering at Universit\'e Laval, where he is pursuing his Master's degree in Artificial Intelligence with a focus on computer vision for robotics.

**Fernando Lopez** is a senior scientist with over 12 years of experience in industry and research in infrared (IR) imaging, advanced non-destructive testing and evaluation (NDT&E) of materials, applied heat transfer, and signal processing. After obtaining his Ph.D. in Mechanical Engineering in 2014, he worked as a Postdoctoral Researcher at Universit'e Laval, conducting research projects with various industrial partners, mainly in aerial IR thermography (IRT) inspection, energy efficiency, and robotic IRT for the NDT&E of aerospace components. He has been the recipient of several academic and research awards, including the 2015 CAPES Doctoral Thesis Award in Engineering, 2015 UFSC Honorable Mention Award, Emergent Leaders of the Americas Award from the Ministry of Foreign Affairs and International Trade of Canada and the Best Presentation Award from 7th International Workshop Advances in Signal Processing for NDE of Materials. Dr. Lopez is currently the Chair of the Program Committee of the CREATE NSERC Innovative Program on NDT and a member of the Standard Council Canada ISO/TC 135/SC 8 on Thermographic Testing. His scientific contributions include more than 20 publications in peer-reviewed journals and international conferences. He is currently Director of Research and Development at TORNGATS, leading several R&D initiatives on advanced NDT&E methods.

**Xavier P.V. Maldague** received the B.Sc., M.Sc., and Ph.D. degrees in electrical engineering from Universite Laval, Quebec City, Canada, in 1982, 1984, and 1989, respectively. He has been a Full Professor with the Department of Electrical and Computing Engineering, Universite Laval, Quebec City, Canada, since 1989, where he was the Head of the Department from 2003 to 2008 and 2018. He has trained over 50 graduate students (M.Sc. and Ph.D.) and has more than 300 publications. His research interests include infrared thermography, nondestructive evaluation (NDE) techniques, and vision/digital systems for industrial inspection. He is an Honorary Fellow of the Indian Society of Nondestructive Testing. He is also a Fellow of the Canadian Engineering Institute, the American Society of Nondestructive Testing, and the Alexander von Humbolt Foundation, Germany. He holds the Tier 1 Canada Research Chair in Infrared Vision. He has been the Chair of the Quantitative Infrared Thermography (QIRT) Council since 2004.

## Contact
Parham Nooralishahi - parham.nooralishahi@gmail.com | [@phm](https://www.linkedin.com/in/parham-nooralishahi/) <br/>

## Acknowledgements
This research is supported by the Canada Research Chair in Multi-polar Infrared Vision (MiViM), the Natural Sciences, and Engineering Research Council (NSERC) of Canada through a discovery grant and by the "oN Duty!" NSERC Collaborative Research and Training Experience (CREATE) Program. Special thanks to TORNGATS company for their support in testing and manufacturing of the parts.
