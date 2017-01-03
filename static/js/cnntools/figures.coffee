
class FigureViewer
  render_figure: (disp_type, figure) ->
    console.log("Rendering #{figure.name} figure...")
    nv.addGraph( =>
      chart = nv.models.lineWithFocusChart()
            .margin({left: 100})  #Adjust chart margins to give the x-axis some breathing room.
            .useInteractiveGuideline(true)  #We want nice looking tooltips and a guideline!
            .showLegend(true)       #Show the legend, allowing users to turn on/off line series.
            .showYAxis(true)        #Show the y-axis
            .showXAxis(true)        #Show the x-axis
            .noData('There is no data for the given filter.')

      chart.xAxis     #Chart x-axis settings
      .axisLabel(figure.xlabel)
      .tickFormat(d3.format(',r'))

      chart.yAxis     #Chart y-axis settings
        .axisLabel(figure.ylabel)
        .tickFormat(d3.format('.3f'))

      d3.select("#chart-#{disp_type}").datum(figure.figure_data).call(chart)
      nv.utils.windowResize(chart.update)
      return chart
    )

$( ->
  window.fig_viewer = new FigureViewer()
  for disp_type, figure of window.figures
    window.fig_viewer.render_figure(disp_type, figure)
)
