// set the dimensions and margins of the graph
const margin = {top: 0, right: 0, bottom: 30, left: 0},
  width = 1150 - margin.left - margin.right,
  height = 450 - margin.top - margin.bottom;

// append the svg object to the body of the page
const svg = d3.select("#my_dataviz").append("svg")
  .attr("width", width + margin.left + margin.right)
  .attr("height", height + margin.top + margin.bottom)

//Read the data
d3.csv("dataset/data_viz.csv").then(function(data) {
    
    
/// PREPARE DATA
    // Add an 'Id' column to the dataset
    data.forEach((d, i) => {
        d.id = i
    })

    // Retrieve the list of features
    const features = Object.keys(data[0]).filter(d => d != "group")
    console.log(features)

    // We will create a data object that will contain three attributes: group (the id), variable (the feature) and value
    const data2 = []
    data.forEach(d => {
        const classValue = d["y"];
        features.forEach(f => {
            if (f != "y"){
            data2.push({group: d.id, variable: f, value: d[f], class: classValue})
            }
        })
    })

    // Sort by classValue
    data2.sort((a, b) => a.class - b.class) // sort by classValue in the ascending order

/// DRAW HEATMAP

  // Labels of row and columns -> unique identifier of the column called 'group' and 'variable'
  const myGroups = Array.from(new Set(data2.map(d => d.group)))
  const myVars = Array.from(new Set(data2.map(d => d.variable)))

  // Build X scales and axis:
  const x = d3.scaleBand()
    .range([ 0, width ])
    .domain(myGroups)
    .padding(0.05);
  svg.append("g")
    .style("font-size", 1)
    .attr("transform", `translate(0, ${height})`)
    .call(d3.axisBottom(x).tickSize(0))
    .select(".domain").remove()

  // Build Y scales and axis:
  const y = d3.scaleBand()
    .range([ height, 0 ])
    .domain(myVars)
    .padding(0.05);
  svg.append("g")
    .style("font-size", 10)
    .call(d3.axisLeft(y).tickSize(0))
    .select(".domain").remove()

  // Build color scale
  const myColor = d3.scaleSequential()
    .interpolator(d3.interpolateViridis)
    .domain([-1.5,1.5])

    // DIsplay the color scale
    svg.append("g")
        .attr("class", "legendSequential")
        .attr("transform", "translate(220,20)")

  // add the squares
  svg.selectAll()
    .data(data2, function(d) {return d.group+':'+d.variable;})
    .join("rect")
      .attr("x", function(d) { return x(d.group) })
      .attr("y", function(d) { return y(d.variable) })
      .attr("rx", 1)
      .attr("ry", 1)
      .attr("width", x.bandwidth() )
      .attr("height", y.bandwidth() )
      .style("fill", function(d) { return myColor(d.value)} )
      .style("stroke-width", 4)
      .style("stroke", "none")
      .style("opacity", 0.8)

    // Spot the first time that the class changes
    let i = 0
    while (i < data2.length && data2[i].class == -1) {
        i++
    }

    // Draw a green vertical line to separate the two classes
    svg.append("line")
        .attr("x1", x(data2[i].group))
        .attr("x2", x(data2[i].group))
        .attr("y1", 0)
        .attr("y2", height)
        .style("stroke", "green")
        .style("stroke-width", 2)
}); 

// Add title to graph
svg.append("text")
        .attr("x", 0)
        .attr("y", -50)
        .attr("text-anchor", "left")
        .style("font-size", "22px")
        .text("A d3.js heatmap");

// Add subtitle to graph
svg.append("text")
        .attr("x", 0)
        .attr("y", -20)
        .attr("text-anchor", "left")
        .style("font-size", "14px")
        .style("fill", "grey")
        .style("max-width", 400)
        .text("A short description of the take-away message of this chart.");