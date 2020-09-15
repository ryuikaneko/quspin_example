set term png
set output "fig_mz0mz1_connected.png"
set xrange [0:2]
set yrange [-0.5:4.5]
set xlabel "Omega/V"
set ylabel "Delta/V"
set cbrange [-0.5:0.3]
p "dat" u 2:3:13 w imag ti ""
