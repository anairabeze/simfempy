// This code was created by PyGmsh v3.0.13.
p4 = newp;
Point(p4) = {0.0, 0.0, 0.0, 0.050000000000000003};
p5 = newp;
Point(p5) = {1.0, 0.0, 0.0, 0.050000000000000003};
p6 = newp;
Point(p6) = {1.0, 1.0, 0.0, 0.050000000000000003};
p7 = newp;
Point(p7) = {0.0, 1.0, 0.0, 0.050000000000000003};
l4 = newl;
Line(l4) = {p4, p5};
l5 = newl;
Line(l5) = {p5, p6};
l6 = newl;
Line(l6) = {p6, p7};
l7 = newl;
Line(l7) = {p7, p4};
ll1 = newll;
Line Loop(ll1) = {l4, l5, l6, l7};
s1 = news;
Plane Surface(s1) = {ll1};
Physical Line(11) = {l4};
Physical Line(22) = {l5};
Physical Line(33) = {l6};
Physical Line(44) = {l7};
Physical Surface(99) = {s1};