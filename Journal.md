- Steps to solve this problem:
    1. Classify leaves and branches or trunks.
    2. How to count leaves? 
        - DBSCAN to count leaves. But each cluster may not represent a leaf.
        - Use RANSAC or Region Growing. To connect disconnected parts of a leaf.
    3. How much occlusion affect the counting? Can it be mitigated?
    4. How to validate the counting results?
        - Use the synthetic data to validate the counting algorithm.
        