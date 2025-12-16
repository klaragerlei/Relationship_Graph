# Trajectory Relationship Graph

MoveApps

Github repository: github.com/klaragerlei/Trajectory-Relationship-Graph

## Description
Visualize animal relationships and group dynamics by analyzing proximity patterns between individuals. This app creates network graphs showing how much time animals spent close to each other, helping researchers understand social structures, group cohesion, and individual associations within tracked populations.

<img width="4178" height="2970" alt="image" src="https://github.com/user-attachments/assets/3c2c247c-0b04-42bc-bd9d-35750024207f" />

## Documentation
The Trajectory Relationship Graph app analyzes movement data from multiple tracked animals to identify and visualize spatial-temporal proximity patterns. The app calculates pairwise distances between all individuals at regular time intervals and counts how many times pairs of animals were within a specified distance threshold of each other. These proximity counts are then used to create a network graph where:

- **Nodes** represent individual animals
- **Edges** (connections) represent relationships between animals
- **Edge thickness** reflects the strength of the relationship (more time spent close = thicker edge)
- **Node colors** can represent different groups (e.g., family units, sex, age class)

The app performs the following analysis steps:

1. **Temporal Alignment**: Resamples all trajectories to a common time step to enable pairwise comparison
2. **Distance Calculation**: Computes distances between all pairs of animals at each time point
3. **Proximity Counting**: Counts how many times each pair was within the meeting distance threshold
4. **Graph Construction**: Builds a network where edge weights represent proximity counts
5. **Edge Filtering**: Removes weak relationships below a specified percentile threshold to highlight meaningful connections
6. **Visualization**: Creates a spring-layout graph with customizable colors, sizes, and labels

The resulting visualization helps identify:
- Core groups and subgroups within the population
- Central vs. peripheral individuals
- Temporal changes in associations (when filtered by year)
- Potential social hierarchies or affiliations

## Application Scope

### Generality of App Usability
This app was developed to work with any taxonomic group where social relationships and proximity patterns are of interest. It is particularly useful for:
- Gregarious species (herding ungulates, social carnivores, flocking birds)
- Species with fission-fusion dynamics
- Studies of family groups, territorial behavior, or mating systems

The app works best when individuals have overlapping home ranges or spend time in shared areas. It may produce limited results for solitary or highly territorial species where proximity events are rare.

### Required Data Properties
- **Multiple individuals**: The app requires data from at least 2 tracked individuals to create relationships
- **Temporal overlap**: Individuals should be tracked during overlapping time periods
- **Regular fix rate**: Optimal results require consistent fix intervals (e.g., 1 location per hour)
- **Sufficient spatial overlap**: Animals should have opportunities to be in proximity for meaningful relationship analysis
- **Group ID attribute** (optional): For colored visualization by groups, a column indicating group membership is recommended

The app performs best with:
- Fix rates between 15 minutes and 2 hours
- Study periods of at least several weeks
- Populations where social interactions occur regularly

## Input Type
`MovingPandas.TrajectoryCollection`

## Output Type
`MovingPandas.TrajectoryCollection` (filtered by year if specified)

## Artefacts
- `relationship_graph.png`: Network graph visualization showing animal relationships. Nodes represent individuals, edges show proximity relationships, edge thickness indicates relationship strength, and node colors indicate group membership (if specified).

## Settings

- **Meeting Distance Threshold** (`meeting-distance`): Distance threshold in meters for considering two animals to be close to each other. When animals are within this distance, it counts as a proximity event. Default: `20.0` meters. Unit: `meters`

- **Time Unit for Resampling** (`time-unit`): Time unit for resampling trajectory data to align all individuals on a common temporal grid. Use pandas datetime frequency strings: 'min' (minutes), 'h' (hours), 'D' (days). Default: `h` (hourly)

- **Group ID Column** (`group-id-column`): Name of the column in your data that contains group identifiers for coloring nodes (e.g., 'group_id', 'sex', 'family', 'species'). Nodes belonging to the same group will be colored identically. Set to empty string to disable group coloring. Default: `group_id`

- **Edge Threshold Percentile** (`edge-threshold-percentile`): Percentile threshold (0-100) for filtering weak edges from the graph. Higher values show fewer, stronger relationships. For example, 40 means only edges above the 40th percentile of proximity counts are shown. Default: `40.0`

- **Node Spacing** (`node-spacing`): Controls how spread out nodes are in the layout. Higher values create more spacing between nodes. Default: `2.0`

- **Minimum Edge Width** (`min-edge-width`): Minimum visual width for edges in the graph. Default: `0.5`

- **Maximum Edge Width** (`max-edge-width`): Maximum visual width for edges in the graph. Default: `3.5`

- **Node Size** (`node-size`): Size of node circles in the visualization. Default: `700`

- **Font Size** (`font-size`): Font size for node labels (if labels are shown). Default: `9`

- **Figure Width** (`figure-width`): Width of output figure in inches. Default: `14`

- **Figure Height** (`figure-height`): Height of output figure in inches. Default: `10`

- **Show Edge Labels** (`show-edge-labels`): Whether to display numeric labels showing proximity counts on edges. Default: `false`

- **Edge Label Threshold Percentile** (`edge-label-threshold-percentile`): Only show labels for edges above this percentile (if Show Edge Labels is enabled). Default: `75.0`

- **Label Strategy** (`label-strategy`): How to display node labels. Options: 'none' (no labels), 'offset' (labels offset from nodes with connecting lines), 'minimal' (only label peripheral nodes to reduce clutter). Default: `none`

- **Filter by Year** (`year`): Optional year to filter the trajectory data before analysis. Only data from this year will be included in the relationship graph. Default: `1960`

## Changes in Output Data
The input TrajectoryCollection is filtered to contain only data from the specified year (if a year is provided). All other trajectory properties remain unchanged. The primary output is the visualization artifact (`relationship_graph.png`), not modifications to the data itself.

## Most Common Errors

1. **No edges in graph**: This occurs when no animals were ever within the meeting distance threshold. Solutions:
   - Increase the `meeting-distance` parameter
   - Check that animals actually overlap spatially in your study
   - Verify that temporal overlap exists between individuals

2. **Graph too cluttered**: When many weak relationships create visual noise. Solutions:
   - Increase `edge-threshold-percentile` to show only strong relationships
   - Increase `node-spacing` to spread out the layout
   - Use `label-strategy: minimal` or `none` to reduce label clutter

3. **Missing year in data**: When filtering by year produces no data. Solution:
   - Verify the year exists in your dataset
   - Check that the timestamp format is correct

4. **Group ID column not found**: When the specified `group-id-column` doesn't exist. Solution:
   - Verify the column name matches exactly (case-sensitive)
   - Set to empty string to disable group coloring

## Null or Error Handling

**Setting `meeting-distance`**: 
- NULL or 0: Defaults to 20.0 meters
- Negative values: Not allowed, will cause error
- Very large values (>10km): May result in all animals being "close" constantly

**Setting `time-unit`**: 
- NULL or invalid: Defaults to 'h' (hours)
- Must be valid pandas frequency string ('min', 'h', 'D', etc.)
- Very fine time resolution (e.g., seconds) may cause memory issues with large datasets

**Setting `group-id-column`**: 
- NULL or empty string: All nodes colored light grey
- Column doesn't exist: Logs warning and colors all nodes grey
- Column contains NULL values: Animals with NULL group are colored grey

**Setting `year`**: 
- NULL: All data is used, no year filtering applied
- Year not in dataset: Returns None and logs warning
- Invalid year: May cause error

**Input data**:
- Less than 2 individuals: Cannot create relationships, returns warning
- No temporal overlap: May result in empty graph with no edges
- Missing coordinates: Rows with NULL coordinates are excluded from distance calculations
- Irregular fix rates: App handles via resampling, but very irregular data may produce less reliable results

**Edge filtering**:
- If `edge-threshold-percentile` removes all edges: Graph shows isolated nodes with no connections
- If only 1-2 individuals remain after filtering: Graph will be very simple with few/no edges
