% This function accepts the location of a resulting table (.csv) from the 
% machine learing algoirthms prefixed with  the name of the algorithm 
% (either "SVM_", "RF_", or "CNN_") called  "test_data". It creates arrays
% contining the different mismatches (true/true, true/false, false/true,
% and false/false). The result is a new table object (.csv) with
% the four concatenated together with the prefix 'mismatched_' saved under 
% the ../analysis_data/ directory.

function getMismatches(location)
data = readtable(location);
data = rmmissing(data);
a = 1;
b = 1;
c = 1;
d = 1;
for i = 1:size(data)
	if (data.output{i} == "True" & data.predicted{i} == "True")
		true_true(a, :) = data(i, :);
		a = a + 1;
	elseif (data.output{i} == "True" & data.predicted{i} == "False")
		true_false(b, :) = data(i, :);
		b = b + 1;
	elseif (data.output{i} == "False" & data.predicted{i} == "True")
		false_true(c, :) = data(i, :);
		c = c + 1;
	elseif (data.output{i} == "False" & data.predicted{i} == "False")
		false_false(d, :) = data(i, :);
		d = d + 1;
	end
end
out = [true_true; true_false; false_true; false_false];
% Changes prefix of the file.
t = char(location);
q = strfind(location, "/");
t(1:q(2)) = '';
loc = string(t);
s = "../analysis_data/mismatched_" + loc;

writetable(out, s, 'Delimiter', ',');

end