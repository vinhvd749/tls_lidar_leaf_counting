
Advanced Methodologies for LiDAR-Based Leaf Counting and Canopy Structure Quantification: A Comprehensive Analysis


1. Introduction

The quantification of foliar elements within vegetation canopies—specifically the counting of leaves and the estimation of leaf area—stands as a central problem in contemporary remote sensing, forestry, and precision agriculture. Leaves function as the primary biological interface for gas exchange, driving the essential processes of photosynthesis, transpiration, and carbon sequestration. Consequently, the ability to derive precise, spatially explicit metrics of leaf distribution is not merely an academic exercise but a fundamental requirement for robust ecological modeling, crop yield forecasting, and the monitoring of forest health in the face of climate change.1
Historically, the assessment of canopy structure relied heavily on direct, destructive sampling or passive optical methods. Destructive techniques, while accurate, are labor-intensive, site-specific, and impossible to scale; optical methods such as hemispherical photography or localized light sensors provide valuable estimates of Leaf Area Index (LAI) but are inherently limited by their two-dimensional nature and saturation issues in dense canopies.3 The emergence of Light Detection and Ranging (LiDAR) technology has fundamentally altered this landscape. As an active remote sensing modality capable of penetrating canopy gaps and recording multiple returns, LiDAR offers a unique capacity to digitize the three-dimensional (3D) architecture of vegetation with millimeter-level precision.5
However, the transition from raw LiDAR point clouds to meaningful biological integers—such as a "leaf count"—is fraught with complexity. It requires navigating the "semantic gap" between geometric coordinates and biological organs. This challenge necessitates a sophisticated interplay of data acquisition strategies, noise filtration, semantic segmentation (differentiating wood from leaf), and instance segmentation (distinguishing individual leaves). This report provides an exhaustive, expert-level examination of the state-of-the-art methods for LiDAR-based leaf counting. It synthesizes geometric and radiometric separation theories, traces the evolution from heuristic clustering to deep learning, and addresses critical environmental factors like wind and occlusion that complicate real-world deployment.

2. Theoretical Foundations of LiDAR-Based Canopy Analysis

Understanding the methodologies for leaf counting requires a deep appreciation of how LiDAR interacts with vegetation. The fundamental premise is that the structural arrangement of points and the intensity of the returned signal encode sufficient information to distinguish foliage from woody material and, subsequently, to isolate individual leaves.

2.1 The Physics of Laser-Canopy Interaction

When a laser pulse intercepts a canopy, it interacts with objects of varying sizes, orientations, and reflectances. The "return" is not just a coordinate in space ($x, y, z$); it is a timestamped signal with an intensity value.
Geometric Scattering: The spatial distribution of returns is dictated by the size of the object relative to the beam footprint. Trunks and large branches, being solid and larger than the footprint, tend to produce coherent, linear, or cylindrical point clusters. Leaves, conversely, are often thin, planar, and randomly oriented, resulting in scattered, planar, or volumetric distributions depending on the scan resolution.5
Radiometric Response: The intensity of the return is a function of the target's reflectivity at the laser's wavelength. Most terrestrial and airborne scanners operate in the near-infrared (NIR, ~1064 nm) or green (~532 nm) wavelengths. At 1064 nm, both healthy vegetation and woody bark have high reflectance, making separation based solely on intensity challenging without calibration. However, multi-spectral systems that include shortwave-infrared (SWIR, ~1550 nm) channels exploit the water absorption features of leaves to create contrast.7

2.2 From Point Clouds to Biological Metrics

The workflow for leaf counting generally follows a hierarchical pipeline:
Pre-processing: Noise removal, registration, and normalization.
Semantic Segmentation (Wood-Leaf Separation): Classifying every point as either "wood" (non-photosynthetic) or "leaf" (photosynthetic).
Instance Segmentation: Grouping "leaf" points into individual distinct leaf instances.
Quantification: Counting the instances or integrating their area.
In scenarios where individual instance segmentation is impossible due to resolution limits (e.g., high-altitude airborne scans), the methodology shifts to voxel-based or gap-fraction approaches that estimate Leaf Area Density (LAD) as a proxy for count.1

3. Data Acquisition Modalities

The selection of a LiDAR platform dictates the resolution of the resulting point cloud and, consequently, the viable algorithmic approaches. The trade-off is invariably between spatial coverage and point density.

3.1 Terrestrial Laser Scanning (TLS)

TLS provides the highest fidelity data for leaf counting. Mounted on a tripod, these systems scan the canopy from below (hemispherical scanning).
Resolution: High-end phase-shift scanners can achieve point spacings of <2 mm at 10 m range. This density is sufficient to resolve the geometry of individual petioles and leaf veins, enabling true geometric primitive fitting.10
Multi-Scan Registration: To mitigate occlusion (shadowing), scans from multiple viewpoints are co-registered. While this increases data volume and processing time, it is essential for accurate counting; single-scan data can underestimate leaf area by 20-40% due to self-occlusion.12
Application: TLS is the standard for developing allometric equations, calibrating airborne data, and conducting detailed plot-level phenotyping.

3.2 Unmanned Aerial Systems (UAS) LiDAR

UAS-LiDAR offers a bridge between ground plots and landscape scales.
Vantage Point: Scanning from above allows for excellent characterization of the upper canopy but often fails to penetrate to the understory in dense forests.
Point Density: Modern sensors (e.g., Riegl miniVUX, DJI L1) can produce densities of 100-500 points/$m^2$. This is often insufficient for counting individual small leaves but adequate for detecting large palm fronds, delineating tree crowns, and estimating voxel-based LAD.13
Integration: UAS platforms frequently carry multispectral cameras alongside LiDAR. The fusion of high-resolution optical imagery (for color-based segmentation) with LiDAR depth is a growing trend for identifying leaves in the upper canopy.13

3.3 Mobile Laser Scanning (MLS) for Phenotyping

In agricultural settings, efficiency is paramount. MLS systems mounted on tractors, gantries, or diverse robotic platforms scan rows of crops continuously.
Consistent Geometry: Unlike hand-held scanning, MLS provides a consistent trajectory, simplifying the registration of "time-of-flight" data.
High Throughput: These systems are critical for "phenomics"—the large-scale collection of phenotypic traits. They generate massive datasets that require automated, real-time processing pipelines to count leaves on thousands of plants per hour.4
Sensor Fusion: MLS units often fuse LiDAR with RGB-D cameras and thermal sensors, providing a multi-modal view that aids in separating fruit from leaves and detecting stress.16

4. Wood-Leaf Separation: The Critical Precursor

Before any leaf can be counted, it must be distinguished from the branch that supports it. "Wood-leaf separation" (or classification) is the most critical semantic segmentation task in the pipeline. Errors here—classifying a dense twig cluster as a leaf or a broad leaf as a branch—propagate directly to the final count. Methodologies fall into three primary domains: Geometric, Radiometric, and Deep Learning.

4.1 Geometric Feature-Based Separation

Geometric methods exploit the spatial arrangement of points. They are attractive because they rely only on $x,y,z$ coordinates, making them applicable to any scanner. The core concept is that wood and leaves exhibit distinct shape descriptors at local scales.

4.1.1 Eigenvalue Decomposition and Shape Descriptors

The standard approach involves computing the structure tensor (covariance matrix) of the $k$-nearest neighbors for every point. The eigenvalues of this matrix ($\lambda_1, \lambda_2, \lambda_3$) describe the spread of points in orthogonal directions. From these, shape features are derived:
Linearity ($L_\lambda = (\lambda_1 - \lambda_2)/\lambda_1$): High for stems, trunks, and branches.
Planarity ($P_\lambda = (\lambda_2 - \lambda_3)/\lambda_1$): High for flat surfaces like broad leaves or ground patches.
Scattering/Sphericity ($S_\lambda = \lambda_3/\lambda_1$): High for complex, volumetric structures like dense foliage clusters where individual leaves are not resolved.5
A critical nuance is that the "optimal" neighborhood size ($k$ or radius $r$) is not constant. A trunk looks linear at $r=50$cm but planar at $r=5$cm (acting like a wall). A petiole looks linear only at very small scales. Therefore, multi-scale features—calculating these descriptors at multiple radii and feeding them into a classifier—are essential for robust separation.18

4.1.2 Graph-Based and Shortest-Path Algorithms

Graph-based methods move beyond local neighborhoods to analyze connectivity.
The Shortest Path Hypothesis: In a connected graph of tree points, the "skeleton" (wood) forms the primary pathways for transport. Geometrically, this means that wood points are central to the structure.
TLSeparation Algorithm: This algorithm constructs a graph and identifies the shortest paths from the base of the tree to all tips. Points residing on these main paths are classified as wood. Leaves are identified as points that are topologically "distal" or disconnected from the main skeleton. While effective for major branches, it struggles with fine twigs, often classifying them as leaves due to their distance from the main trunk.19
LeWoS (Leaf-Wood Separation): This recursive cut algorithm first segments the cloud into clusters and then evaluates the "linearity" and size of each cluster. Large, linear clusters are solidified as wood. The algorithm then "grows" out from these wood seeds, re-evaluating adjacent clusters. This recursive approach allows it to capture smaller branches, but it is computationally intensive and sensitive to the initial clustering parameters.11
CWLS (Connectivity-based Wood-Leaf Separation): A newer algorithm, CWLS, integrates geometric classification with connectivity analysis. It has demonstrated superior performance over LeWoS by effectively reducing noise while maintaining branch continuity, particularly for higher-order branches in the upper canopy.23

4.1.3 Unsupervised Geometric Segmentation

Some approaches avoid training classifiers altogether. For example, the TLSeparation library and related methods often use unsupervised clustering (like Gaussian Mixture Models) on the geometric features. They assume that the feature space naturally separates into two modes (linear/wood and planar/scattered/leaf).24 However, in complex forests with lianas, dead wood, and diverse leaf shapes, this bimodal assumption often fails.

4.2 Radiometric and Multispectral Separation

Intensity data adds a layer of physical discrimination independent of geometry.

4.2.1 Single-Wavelength Intensity

Standard NIR LiDAR intensity is influenced by range, incidence angle, and target reflectance. Even after correcting for range (using the radar equation), separation is difficult because both wet bark and leaves can have similar reflectance at 1064 nm. However, at 532 nm (green), woody material is often more reflective than chlorophyll-absorbing leaves, offering some contrast.8

4.2.2 Multispectral and Dual-Wavelength LiDAR

The most robust radiometric separation comes from multispectral systems.
Mechanism: These systems fire lasers at multiple wavelengths (e.g., 1064 nm and 1550 nm).
The Water Absorption Feature: The 1550 nm (SWIR) wavelength is strongly absorbed by liquid water. Leaves, being water-rich, appear distinctively "darker" (lower intensity) in SWIR channels compared to woody branches, which have lower water content and higher reflectance.
Normalized Difference Indices: By calculating an index similar to NDVI (e.g., $\frac{I_{1064} - I_{1550}}{I_{1064} + I_{1550}}$), researchers can create a highly separable feature space. This method is particularly valuable because it works on a point-by-point basis, independent of local point density.7
Limitations: The primary limitation is hardware availability and cost. Additionally, surface wetness (rain/dew) can confound the water absorption signal, making leaves and wet branches look similar.

4.3 Supervised Learning for Classification

The current state-of-the-art involves fusing geometric and radiometric features into supervised machine learning models.
Classifiers: Random Forest (RF), Support Vector Machines (SVM), and Gradient Boosting Machines (XGBoost) are commonly used.
Feature Engineering: The input vector for each point typically includes:
Geometric features at 3-5 scales (Linearity, Planarity, Sphericity).
Verticality (Z-range in neighborhood).
Intensity and Pulse Width (if available).
Return Number (leaves often produce multiple returns; trunks produce single returns).
Performance: RF models generally achieve accuracies of ~94% when trained on representative data. The key advantage of RF is its ability to handle high-dimensional data and effectively weight features; for instance, it might learn that "linearity at 50cm" is the most predictive feature for trunks, while "intensity" is most predictive for leaves.11

4.4 Deep Learning Approaches

Deep learning bypasses manual feature engineering, learning optimal descriptors directly from the data.
PointNet++: This architecture consumes raw point clouds. By applying Multi-Layer Perceptrons (MLPs) to local neighborhoods and aggregating features hierarchically, PointNet++ learns to recognize the "texture" of leaves versus wood. It has shown high success in semantic segmentation tasks, often serving as the backbone for crop phenotyping pipelines.16
Deep Learning + Multispectral: Recent work suggests that adding multispectral channels ($x, y, z, I_{1064}, I_{1550}$) to Deep Learning inputs significantly boosts performance, particularly for fine branches.29

5. Instance Segmentation: The Mechanics of Counting

Once the point cloud is semantically segmented (i.e., we know which points are "leaf"), the task shifts to instance segmentation: determining that this group of points is Leaf #1 and that group is Leaf #2. This is the core "counting" step.

5.1 Clustering Algorithms

Clustering assumes that points belonging to a single leaf are spatially contiguous and separated from other leaves by a gap or a change in density.

5.1.1 DBSCAN and Adaptive Variants

Density-Based Spatial Clustering of Applications with Noise (DBSCAN) is the workhorse of leaf clustering.
Core Logic: It groups points that are density-reachable from one another. If a point has at least $minPts$ neighbors within radius $\epsilon$, a cluster is formed.
The Challenge: Vegetation density is non-uniform. Points near the scanner are dense; points at the top of the canopy are sparse. A fixed $\epsilon$ will either over-segment the dense areas (splitting one leaf into many) or under-segment sparse areas (merging distinctive leaves).
Adaptive Solutions: Adaptive DBSCAN (A-DBSCAN) algorithms dynamically adjust $\epsilon$ based on the local point density or the range from the scanner. This ensures consistent segmentation performance across the entire vertical profile of the tree.28
Performance: Adaptive DBSCAN has been shown to improve precision/recall by ~5-10% over static DBSCAN in variable-density forest scans.30

5.1.2 Mean Shift and Mode Seeking

Mean Shift clustering is a non-parametric iterative algorithm.
Mechanism: It does not require a pre-defined number of clusters. For every point, it computes the mean of neighbors within a bandwidth and shifts the "center" to that mean. It repeats this until it converges on a mode (density maximum).
Application: In leaf counting, Mean Shift is excellent for finding the center of leaf clusters or rosettes. It is often initialized with "seed points" derived from local curvature maxima.
Comparison: Mean Shift is generally more robust to irregular shapes than K-Means but is computationally expensive ($O(N^2)$ per iteration) compared to DBSCAN.34

5.2 Geometric Primitive Fitting (RANSAC)

For crops with large, distinct, and relatively flat leaves (e.g., maize, pumpkin, cucumber), geometric model fitting is highly effective.
RANSAC (Random Sample Consensus): The algorithm randomly selects a subset of points (e.g., 3 points) to define a plane. It then counts how many other points fit this plane within a tolerance. The best-fitting plane is extracted as a "leaf," and the process repeats.
Refinements:
NDT-RANSAC: Uses Normal Distributions Transform to pre-cluster points, speeding up the fitting.
Constraint Checking: To avoid fitting a plane across two spatially separated leaves (which might be coplanar), algorithms check for connectivity and surface normal consistency.37
Limitations: This method fails for curled, twisted, or needle-like leaves.

5.3 Region Growing and Watershed

These methods treat the point cloud surface like a terrain.
Region Growing: Starts from a high-confidence seed point (e.g., the center of a planar patch) and adds neighbors if they share similar properties (e.g., normal vector deviation < 10 degrees). This is intuitive for tracing a leaf surface from the midrib out to the edges.32
Watershed: Transforms the point cloud density into a "height map." The "basins" represent individual leaves, and the "ridges" represent the boundaries between overlapping leaves. The challenge lies in defining the correct scale; over-segmentation is common.42

5.4 Deep Learning for Instance Segmentation

The most advanced counting methods now utilize deep neural networks designed for 3D instance segmentation.

5.4.1 TreeLearn and PointGroup

Architectures like TreeLearn (originally for tree separation) and PointGroup use a dual-branch approach. One branch predicts semantic labels (leaf/wood), and the other predicts offset vectors that shift every point toward its instance center. Clustering is then performed on these shifted points, which are much tighter and easier to separate than the original cloud.43

5.4.2 Unsupervised Deep Learning: GrowSP

Annotating 3D point clouds for training is expensive. GrowSP (Growing Superpoints) is a novel unsupervised method.
Concept: It starts by over-segmenting the cloud into "superpoints" (small, geometrically homogeneous patches). A network then learns to merge these superpoints based on semantic similarity rules derived from the data structure itself, effectively "discovering" leaf instances without explicit labels.
Significance: This is a breakthrough for scaling leaf counting to new species where no training data exists.29

5.4.3 Graph Neural Networks (GNNs)

SG-Net and similar GNNs construct a graph where edges represent spatial and feature similarity. By performing graph convolution, the network learns the topology of the plant. This is crucial for distinguishing a leaf touching a stem (spatial proximity) from a leaf growing from a stem (structural connectivity).46

6. Indirect Quantification: Voxel-Based Approaches

In scenarios where individual leaves cannot be resolved (e.g., airborne scans of dense forests, or high-throughput phenotyping of wheat), counting is replaced by estimating Leaf Area Density (LAD).

6.1 Voxelization Mechanics

The point cloud is discretized into a 3D grid of voxels (volumetric pixels).
Attribute: Each voxel tracks statistics like the number of returns, intensity mean, or presence/absence.
Voxel Size Sensitivity: The choice of voxel size is the single most critical parameter. If too large (> leaf size), multiple leaves are aggregated, leading to underestimation of area. If too small (< beam footprint), a single leaf is fragmented, leading to overestimation. Studies suggest an optimal voxel size is often related to the beam footprint or average leaf characteristic dimension (e.g., 5-10 cm for trees).9

6.2 LAD Estimation Formulas

The estimation of Leaf Area Density ($LAD$, in $m^2/m^3$) for a voxel layer relies on the contact frequency of laser beams. It essentially inverts the probability of a beam passing through the voxel without interception.
The widely used voxel-based Beer-Lambert law adaptation is:


$$LAD(h, \Delta H) = \frac{1}{\Delta H} \sum_{k} \alpha(\theta) \frac{n_I(k)}{n_I(k) + n_P(k)}$$
Where:
$\Delta H$: Layer thickness.
$n_I(k)$: Number of laser beams intercepted by the voxel.
$n_P(k)$: Number of laser beams passing through the voxel.
$\alpha(\theta)$: Correction coefficient for the angle of incidence and leaf angle distribution ($G$-function).
This method provides a vertical profile of foliage area, which can be integrated to get the total Leaf Area Index (LAI) or divided by an average leaf size to approximate a count.47

6.3 Gap Fraction Inversion

A parallel approach uses Gap Fraction Theory ($P(\theta)$). Instead of counting intercepts per voxel, it looks at the overall probability of gaps at different zenith angles.
Inversion: $L = -\frac{\ln(P(\theta)) \cos(\theta)}{G(\theta) \Omega}$.
Clumping Index ($\Omega$): A critical correction factor. Leaves in nature are not randomly distributed; they are clumped in whorls and branches. Ignoring clumping (assuming $\Omega=1$) leads to significant underestimation of LAI (often by 30-50%). LiDAR's ability to measure gap size distribution allows for the direct estimation of $\Omega$, making it superior to passive optical sensors.2

7. Handling Environmental Challenges: Occlusion and Wind

Real-world LiDAR data is rarely perfect. Two major physical factors—occlusion and wind—can severely degrade counting accuracy and require specialized algorithmic handling.

7.1 Occlusion: The Hidden Canopy

In dense canopies, the "front" leaves block the laser from reaching the "back" leaves.
Impact: This leads to incomplete point clouds and systematic undercounting.
Symmetry-Based Correction: For isolated plants (e.g., in a phenotyping chamber), algorithms may assume radial symmetry. If the scanner sees only the front 180 degrees, the count is simply doubled. This is heuristic and fails for asymmetric plants.
Deep Completion (Inpainting): The cutting edge solution is Point Cloud Completion. Networks like Point Fractal Networks (PFCN) or Tree Completion Net (TC-Net) are trained to take a partial, occluded point cloud and "hallucinate" the missing geometry.
Mechanism: They use an encoder-decoder structure. The encoder condenses the visible features; the decoder generates a complete point cloud that is topologically consistent with the input.
Result: These networks can restore the shape of occluded leaves, allowing for accurate area calculation and counting even when 30-50% of the data is missing.53
Multi-View Fusion: Combining data from ground robots (UGV) and drones (UAV) provides distinct view angles, minimizing blind spots. Probabilistic voxel grids (e.g., OctoMap) are used to fuse these inputs, marking voxels as "occupied," "free," or "unknown".16

7.2 Wind: The Dynamic Noise

Wind causes leaves to flutter or branches to sway during the scan.
Artifacts: A moving leaf creates "ghost points" or a blurred cloud, which geometric classifiers often misinterpret as "scattered" volumetric noise rather than a planar leaf. It also breaks the connectivity required for graph-based separation.
Correction Strategies:
4D LiDAR: High-frequency scanners track the motion. By averaging positions over time, the "rest position" of the leaf can be estimated.
Feature Selection: Research using simulators (like HELIOS++) shows that under windy conditions, fine-scale curvature features become unreliable. Robust classifiers should shift weight toward intensity-based features or larger-scale geometric descriptors which are less sensitive to small displacements.12
Dynamic Training: Training deep learning models on "windy" synthetic data (augmented with noise/displacement) improves their robustness on real-world dynamic datasets.12

8. Application: Agricultural Phenotyping

In agriculture, the goal is High-Throughput Phenotyping (HTP). The requirements here are speed and instance-level precision for breeding and yield prediction.
Crop-Specific Challenges: Crops like maize and sorghum have long, overlapping, arching leaves.
Skeletonization Algorithms: A common technique is to reduce the plant to a 1D curve skeleton.
Method: The algorithm extracts the centerline of the point cloud. Branches of the skeleton connected to the main stem are counted as leaves.
Refinement: Graph-based analysis differentiates between the "main stem" path and "leaf" paths based on verticality and length.56
Leaf Tip Detection: For linear leaves, detecting local maxima in geodesic distance from the stem base is a robust proxy for leaf count. Even if the leaf base is occluded, the tip is often visible.56
Deep Learning Integration: Models like YOLO (You Only Look Once) adapted for projected 2D views or 3D equivalents are used to detect leaf instances. Validation against manual counts typically shows high correlation ($R^2 > 0.95$) for seedlings, though accuracy drops as canopy closure increases.57

9. Comparative Analysis of Methodologies

The following table summarizes the primary methodologies, highlighting their mechanisms, strengths, and ideal use cases.
Methodology
Core Mechanism
Key Advantages
Limitations
Best Application
Geometric (TLSeparation)
Shortest path, Eigenvalues
Sensor agnostic; No training data required
Fails on fine twigs; Sensitive to point density
Structural modeling of individual trees
Graph-Based (LeWoS)
Recursive graph segmentation
Captures branching structure well
Conservative (misses small leaves); Slow
Forest plot wood-leaf separation
Clustering (Adaptive DBSCAN)
Density reachability
Handles arbitrary shapes; Robust to noise
Parameter sensitivity; Computationally heavy
Instance segmentation of leaves in plots
Radiometric (Dual-Wavelength)
Water absorption (1064/1550nm)
High physical contrast; Fast
Requires specialized hardware; Wetness issues
Wood-leaf separation in diverse forests
Deep Learning (PointNet++)
Learned features (Raw points)
High accuracy; Context aware
Data hungry (needs labels); "Black box"
Semantic segmentation of complex crops/trees
Unsupervised Deep Learning (GrowSP)
Superpoint merging
No labeling needed; Scalable
Newer, less proven than supervised methods
Large-scale forest inventory
Voxel-Based (LAD)
Contact frequency / Gap fraction
Efficient; Handles occlusion statistically
Resolution bias; Does not provide integer count
LAI estimation; Biomass quantification
Primitive Fitting (RANSAC)
Geometric model (Plane)
High precision for flat leaves
Fails on curved/complex leaves
Crop phenotyping (Maize, Pumpkin)


10. Conclusion and Future Outlook

The counting of leaves using LiDAR data has matured from a theoretical possibility to a practical reality, driven by the convergence of high-resolution hardware and advanced computational intelligence. The field has moved beyond simple gap-fraction proxies to true instance segmentation, enabling the digital reconstruction of individual plants.
Key Takeaways:
Hybridization is Key: No single method is perfect. The most robust pipelines today utilize hybrid approaches: combining radiometric data (for initial separation) with geometric features (for refinement) and deep learning (for complex pattern recognition).
The Role of AI: Deep learning, particularly unsupervised methods like GrowSP and completion networks like PFCN, addresses the two biggest bottlenecks: the cost of data annotation and the problem of occlusion. This represents the future direction of the field.
Environmental Awareness: Algorithms are no longer static. The integration of wind correction and adaptive density parameters reflects a growing maturity in handling real-world, messy environmental data.
As LiDAR technology continues to miniaturize and integrate with other spectral sensors, the granularity of canopy analysis will only increase. We are approaching a future where "digital forests" are not just statistical abstractions but true twin models, where every leaf is counted, measured, and monitored.
1
Works cited
Estimation of LAI with the LiDAR Technology: A Review - ResearchGate, accessed November 20, 2025, https://www.researchgate.net/publication/345999847_Estimation_of_LAI_with_the_LiDAR_Technology_A_Review
Estimation of Forest LAI Using Discrete Airborne LiDAR: A Review - MDPI, accessed November 20, 2025, https://www.mdpi.com/2072-4292/13/12/2408
What Affects Leaf Area Index Estimation Accuracy in Field and Remote Methods?, accessed November 20, 2025, https://cid-inc.com/blog/what-affects-leaf-area-index-estimation-accuracy-in-field-and-remote-methods/
Estimating Leaf Area Index in Row Crops Using Wheel-Based and Airborne Discrete Return Light Detection and Ranging Data - PMC - PubMed Central, accessed November 20, 2025, https://pmc.ncbi.nlm.nih.gov/articles/PMC8667472/
A Geometric Method for Wood-Leaf Separation Using Terrestrial and Simulated Lidar Data - ASPRS, accessed November 20, 2025, https://www.asprs.org/a/publications/pers/2015journals/PERS_October_2015_Public/HTML/files/assets/common/downloads/page0017.pdf
(PDF) Geometric Primitives in LiDAR Point Clouds: A Review - ResearchGate, accessed November 20, 2025, https://www.researchgate.net/publication/338957023_Geometric_Primitives_in_LiDAR_Point_Clouds_A_Review
Feasibility Study of Wood-Leaf Separation Based on Hyperspectral LiDAR Technology in Indoor Circumstances - IEEE Xplore, accessed November 20, 2025, https://ieeexplore.ieee.org/document/9648015/
Full article: Multisensor and Multispectral LiDAR Characterization and Classification of a Forest Environment - Taylor & Francis Online, accessed November 20, 2025, https://www.tandfonline.com/doi/full/10.1080/07038992.2016.1196584
LiDAR Voxel-Size Optimization for Canopy Gap Estimation - MDPI, accessed November 20, 2025, https://www.mdpi.com/2072-4292/14/5/1054
leaf separation from terrestrial laser scanning point clouds at the forest plot level - University of Twente Research Information, accessed November 20, 2025, https://research.utwente.nl/files/266691705/2041_210X.13715.pdf
Cluster-Based Wood–Leaf Separation Method for Forest Plots Using Terrestrial Laser Scanning Data - MDPI, accessed November 20, 2025, https://www.mdpi.com/2072-4292/16/18/3355
Wind during terrestrial laser scanning of trees: Simulation-based assessment of effects on point cloud features and leaf-wood, accessed November 20, 2025, https://isprs-annals.copernicus.org/articles/X-G-2025/25/2025/isprs-annals-X-G-2025-25-2025.pdf
Evaluation of Leaf Area Index (LAI) of Broadacre Crops Using UAS-Based LiDAR Point Clouds and Multispectral Imagery - IEEE Xplore, accessed November 20, 2025, https://ieeexplore.ieee.org/document/9769924/
Drone LiDAR Occlusion Analysis and Simulation from Retrieved Pathways to Improve Ground Mapping of Forested Environments - MDPI, accessed November 20, 2025, https://www.mdpi.com/2504-446X/9/2/135
Leaf Counting in Top-Down Crop Images Under Occlusion: A Literature Review for Vision-Based Agricultural Monitoring - Appropedia, accessed November 20, 2025, https://www.appropedia.org/Leaf_Counting_in_Top-Down_Crop_Images_Under_Occlusion:_A_Literature_Review_for_Vision-Based_Agricultural_Monitoring
Segment Any Leaf 3D: A Zero-Shot 3D Leaf Instance Segmentation ..., accessed November 20, 2025, https://pmc.ncbi.nlm.nih.gov/articles/PMC11769372/
(PDF) Comparative Analysis of Machine Learning and Deep Learning Models for Individual Tree Structure Segmentation Using Terrestrial LiDAR Point Cloud Data - ResearchGate, accessed November 20, 2025, https://www.researchgate.net/publication/393217363_Comparative_Analysis_of_Machine_Learning_and_Deep_Learning_Models_for_Individual_Tree_Structure_Segmentation_Using_Terrestrial_LiDAR_Point_Cloud_Data
Improved Supervised Learning-Based Approach for Leaf and Wood ..., accessed November 20, 2025, https://ieeexplore.ieee.org/document/8889474/
A connectivity-based algorithm for wood–leaf separation from terrestrial laser scanning data - University of Helsinki Research Portal, accessed November 20, 2025, https://researchportal.helsinki.fi/en/publications/a-connectivity-based-algorithm-for-woodleaf-separation-from-terre/
Leaf and Wood Classification in Southern Pines Trees Using High Resolution Terrestrial Laser Scanning Data - IEEE Xplore, accessed November 20, 2025, https://ieeexplore.ieee.org/document/10282925/
Leaf and wood classification framework for terrestrial LiDAR point clouds - UCL Discovery, accessed November 20, 2025, https://discovery.ucl.ac.uk/10068584/1/Disney_Leaf%20and%20wood%20classification%20framework%20for%20terrestrial%20LiDAR%20point%20clouds_Proof.pdf
Graph-Based Leaf–Wood Separation Method for Individual Trees Using Terrestrial Lidar Point Clouds - ResearchGate, accessed November 20, 2025, https://www.researchgate.net/publication/365106625_Graph-based_Leaf-Wood_Separation_Method_for_Individual_Trees_Using_Terrestrial_Lidar_Point_Clouds
(PDF) A connectivity‐based algorithm for wood–leaf separation from terrestrial laser scanning data - ResearchGate, accessed November 20, 2025, https://www.researchgate.net/publication/396866290_A_connectivity-based_algorithm_for_wood-leaf_separation_from_terrestrial_laser_scanning_data
Intercomparison of methods for estimating leaf inclination angle distribution with terrestrial lidar for broadleaf tree species - PMC - NIH, accessed November 20, 2025, https://pmc.ncbi.nlm.nih.gov/articles/PMC12409099/
Separating leaves from trunks and branches with dual-wavelength terrestrial lidar scanning | Request PDF - ResearchGate, accessed November 20, 2025, https://www.researchgate.net/publication/271426282_Separating_leaves_from_trunks_and_branches_with_dual-wavelength_terrestrial_lidar_scanning
Can Leaf Water Content Be Estimated Using Multispectral Terrestrial Laser Scanning? A Case Study With Norway Spruce Seedlings - PMC - PubMed Central, accessed November 20, 2025, https://pmc.ncbi.nlm.nih.gov/articles/PMC5853165/
Wood–Leaf Classification of Tree Point Cloud Based on Intensity and Geometric Information, accessed November 20, 2025, https://www.mdpi.com/2072-4292/13/20/4050
An automated approach for wood-leaf separation from terrestrial LIDAR point clouds using the density based clustering algorithm DBSCAN | Request PDF - ResearchGate, accessed November 20, 2025, https://www.researchgate.net/publication/325352166_An_automated_approach_for_wood-leaf_separation_from_terrestrial_LIDAR_point_clouds_using_the_density_based_clustering_algorithm_DBSCAN
Unsupervised deep learning for semantic segmentation of multispectral LiDAR forest point clouds - arXiv, accessed November 20, 2025, https://arxiv.org/html/2502.06227v1
Segmenting Individual Tree from TLS Point Clouds Using Improved DBSCAN - MDPI, accessed November 20, 2025, https://www.mdpi.com/1999-4907/13/4/566
(PDF) An Improved DBSCAN Method for LiDAR Data Segmentation with Automatic Eps Estimation - ResearchGate, accessed November 20, 2025, https://www.researchgate.net/publication/330248426_An_Improved_DBSCAN_Method_for_LiDAR_Data_Segmentation_with_Automatic_Eps_Estimation
An Improved DBSCAN Method for LiDAR Data Segmentation with Automatic Eps Estimation, accessed November 20, 2025, https://pmc.ncbi.nlm.nih.gov/articles/PMC6338962/
An Efficient Class-Constrained DBSCAN Approach for Large-scale Point Cloud Clustering - IEEE Xplore, accessed November 20, 2025, https://ieeexplore.ieee.org/iel7/4609443/4609444/09868135.pdf
A New Strategy for Individual Tree Detection and Segmentation from Leaf-on and Leaf-off UAV-LiDAR Point Clouds Based on Automatic Detection of Seed Points - MDPI, accessed November 20, 2025, https://www.mdpi.com/2072-4292/15/6/1619
Automatic Detection of Urban Trees from LiDAR Data Using DBSCAN and Mean Shift Clustering Methods in Fatih, Istanbul, accessed November 20, 2025, https://isprs-archives.copernicus.org/articles/XLVIII-M-6-2025/95/2025/isprs-archives-XLVIII-M-6-2025-95-2025.pdf
An Automatic Hierarchical Clustering Method for the LiDAR Point Cloud Segmentation of Buildings via Shape Classification and Outliers Reassignment - MDPI, accessed November 20, 2025, https://www.mdpi.com/2072-4292/15/9/2432
[1811.08988] Supervised Fitting of Geometric Primitives to 3D Point Clouds - arXiv, accessed November 20, 2025, https://arxiv.org/abs/1811.08988
An Improved RANSAC for 3D Point Cloud Plane Segmentation Based on Normal Distribution Transformation Cells - MDPI, accessed November 20, 2025, https://www.mdpi.com/2072-4292/9/5/433
3D RANSAC Algorithm for Lidar PCD Segmentation | by AJith RaJ - Medium, accessed November 20, 2025, https://medium.com/@ajithraj_gangadharan/3d-ransac-algorithm-for-lidar-pcd-segmentation-315d2a51351
On Enhancing Ground Surface Detection from Sparse Lidar Point Cloud - arXiv, accessed November 20, 2025, https://arxiv.org/pdf/2105.11649
An Overlapping-Free Leaf Segmentation Method for Plant Point Clouds - IEEE Xplore, accessed November 20, 2025, https://ieeexplore.ieee.org/iel7/6287639/8600701/08830350.pdf
Individual-Tree Segmentation and Extraction based on LiDAR Point Cloud Data - International Journal on Advanced Science, Engineering and Information Technology, accessed November 20, 2025, https://ijaseit.insightsociety.org/index.php/ijaseit/article/download/11332/4461/49019
[2309.08471] TreeLearn: A deep learning method for segmenting individual trees from ground-based LiDAR forest point clouds - arXiv, accessed November 20, 2025, https://arxiv.org/abs/2309.08471
TreeLearn: A Comprehensive Deep Learning Method for Segmenting Individual Trees from Ground-Based LiDAR Forest Point Clouds - arXiv, accessed November 20, 2025, https://arxiv.org/html/2309.08471v2
GrowSP: Unsupervised Semantic Segmentation of 3D Point Clouds - CVF Open Access, accessed November 20, 2025, https://openaccess.thecvf.com/content/CVPR2023/papers/Zhang_GrowSP_Unsupervised_Semantic_Segmentation_of_3D_Point_Clouds_CVPR_2023_paper.pdf
Banana Individual Segmentation and Phenotypic Parameter Measurements Using Deep Learning and Terrestrial LiDAR - IEEE Xplore, accessed November 20, 2025, https://ieeexplore.ieee.org/iel7/6287639/10380310/10491195.pdf
Estimating Leaf Area Density of Individual Trees Using the Point Cloud Segmentation of Terrestrial LiDAR Data and a Voxel-Based Model - MDPI, accessed November 20, 2025, https://www.mdpi.com/2072-4292/9/11/1202
Estimation of the leaf area density distribution of individual trees using high-resolution and multi-return airborne LiDAR data - arXiv, accessed November 20, 2025, https://arxiv.org/pdf/2206.11479
Voxel-Based 3-D Modeling of Individual Trees for Estimating Leaf Area Density Using High-Resolution Portable Scanning Lidar - ResearchGate, accessed November 20, 2025, https://www.researchgate.net/publication/3204307_Voxel-Based_3-D_Modeling_of_Individual_Trees_for_Estimating_Leaf_Area_Density_Using_High-Resolution_Portable_Scanning_Lidar
A photographic gap fraction method for estimating leaf area of isolated trees: Assessment with 3D digitized plants - PubMed, accessed November 20, 2025, https://pubmed.ncbi.nlm.nih.gov/16740488/
Plant canopy gap-size analysis theory for improving optical measurements of leaf-area index - Optica Publishing Group, accessed November 20, 2025, https://opg.optica.org/ao/fulltext.cfm?uri=ao-34-27-6211
Retrieving Leaf Area Index (LAI) Using Remote Sensing: Theories, Methods and Sensors - PMC - PubMed Central, accessed November 20, 2025, https://pmc.ncbi.nlm.nih.gov/articles/PMC3348792/
Point Cloud Completion of Plant Leaves under Occlusion Conditions Based on Deep Learning - PubMed, accessed November 20, 2025, https://pubmed.ncbi.nlm.nih.gov/38239737/
(PDF) Point Cloud Completion of Plant Leaves under Occlusion Conditions Based on Deep Learning - ResearchGate, accessed November 20, 2025, https://www.researchgate.net/publication/375313419_Point_cloud_completion_of_plant_leaves_under_occlusion_conditions_based_on_deep_learning
Tree Completion Net: A Novel Vegetation Point Clouds Completion Model Based on Deep Learning - MDPI, accessed November 20, 2025, https://www.mdpi.com/2072-4292/16/20/3763
A graph-based approach for simultaneous semantic and instance segmentation of plant 3D point clouds - Frontiers, accessed November 20, 2025, https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2022.1012669/full
Leaf area estimation in small-seeded broccoli using a lightweight instance segmentation framework based on improved YOLOv11-AreaNet - Frontiers, accessed November 20, 2025, https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2025.1622713/full
A Segmentation-Guided Deep Learning Framework for Leaf Counting - Frontiers, accessed November 20, 2025, https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2022.844522/full
[2502.06227] Unsupervised deep learning for semantic segmentation of multispectral LiDAR forest point clouds - arXiv, accessed November 20, 2025, https://arxiv.org/abs/2502.06227
Can lidars assess wind plant blockage in simple terrain? A WRF-LES study - AIP Publishing, accessed November 20, 2025, https://pubs.aip.org/aip/jrse/article/14/6/063303/2848666/Can-lidars-assess-wind-plant-blockage-in-simple
Plant stem and leaf segmentation and phenotypic parameter extraction using neural radiance fields and lightweight point cloud segmentation networks - Frontiers, accessed November 20, 2025, https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2025.1491170/full
Leaf Area Estimation by Semantic Segmentation ... - CVF Open Access, accessed November 20, 2025, https://openaccess.thecvf.com/content/ICCV2021W/CVPPA/papers/Masuda_Leaf_Area_Estimation_by_Semantic_Segmentation_of_Point_Cloud_of_ICCVW_2021_paper.pdf
A Zero-Shot 3D Leaf Instance Segmentation Method Based on Multi-View Images - MDPI, accessed November 20, 2025, https://www.mdpi.com/1424-8220/25/2/526
An Overlapping-Free Leaf Segmentation Method for Plant Point Clouds - IEEE Xplore, accessed November 20, 2025, https://ieeexplore.ieee.org/document/8830350/
Occluded Apple Fruit Detection and Localization with a Frustum-Based Point-Cloud-Processing Approach for Robotic Harvesting - MDPI, accessed November 20, 2025, https://www.mdpi.com/2072-4292/14/3/482
(PDF) Real-Time Plant Leaf Counting Using Deep Object Detection Networks, accessed November 20, 2025, https://www.researchgate.net/publication/348248970_Real-Time_Plant_Leaf_Counting_Using_Deep_Object_Detection_Networks
A Direction-Adaptive DBSCAN-Based Method for Denoising ICESat-2 Photon Point Clouds in Forested Environments - MDPI, accessed November 20, 2025, https://www.mdpi.com/1999-4907/16/3/524
(PDF) THE EFFECT OF WIND ON TREE STEM PARAMETER ESTIMATION USING TERRESTRIAL LASER SCANNING - ResearchGate, accessed November 20, 2025, https://www.researchgate.net/publication/303843289_THE_EFFECT_OF_WIND_ON_TREE_STEM_PARAMETER_ESTIMATION_USING_TERRESTRIAL_LASER_SCANNING
Simulating Wind Disturbances over Rubber Trees with Phenotypic Trait Analysis Using Terrestrial Laser Scanning - MDPI, accessed November 20, 2025, https://www.mdpi.com/1999-4907/13/8/1298
LiDARPheno – A Low-Cost LiDAR-Based 3D Scanning System for Leaf Morphological Trait Extraction - Frontiers, accessed November 20, 2025, https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2019.00147/full
