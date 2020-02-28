% This function accepts the location of mismatched tables (.csv) prefixed
% with 'mismatched_' and a feature from the data set and plots the
% distrubition of the feature across both true mismatches and false
% mismatches.

function data1DPlot (location, feature)
data = readtable(location);
data = rmmissing(data);
s = height(data);
data1 = [];
data2 = [];

% DO THE SAME FOR FALSE MISMATCHES

for i = 1:s
	if data.output{i} == "TRUE" & data.test{i} == "TRUE"
		data1 = [data1 data.(feature)(i)];
	end
	if data.output{i} == "TRUE" & data.test{i} == "FALSE"
		data2 = [data2 data.(feature)(i)];
	end
end

% DISTRIBUTION FOR NON-BASE FEATURES

edges = [0 0.26 0.51 0.76 1.0];
h = histogram(data1, edges, 'Normalization', 'probability'); % data1 = red
h.FaceColor = [1 0 0];
hold on
h = histogram(data2, edges, 'Normalization', 'probability'); % data2 = blue
h.FaceColor = [0 0 1];
grid on;

t = char(feature);
t(strfind(feature, "_")) = ' ';
feature = string(t);

title("Distribution of " + feature, 'FontSize', 18);
xlabel(feature, 'FontSize', 14);
ylabel('Occurance', 'FontSize', 14);
legend('Predicted True/True','Predicted False/True')
xticks([.13 .375 .635 .88])
set(gca,'xticklabel',{'A','T','C', 'G'});
hold off
saveas(h, "../features/" + feature + ".png");
