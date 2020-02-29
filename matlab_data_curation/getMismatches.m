% This function accepts the location of a resulting table (.csv) from the 
% machine learing algoirthms prefixed with "examine_" and the name of the 
% algorithm (either "SVM_", "RF_", or "CNN_"). It creates four arrays  
% contining the different mismatches (true/true, true/false, false/true,
% and false/false). The result is a new table object (.csv) with
% the four concatenated together with the prefix 'mismatched_' saved under 
% the ../data/ directory.

function getMismatches(location)
data = readtable(location);
data = rmmissing(data);
a = 1;
b = 1;
c = 1;
d = 1;
for i = 1:size(data)
	if (data.output{i} == "True" & data.predict{i} == "True")
		true_true(a, :) = data(i, :);
		a = a + 1;
	elseif (data.output{i} == "True" & data.predict{i} == "False")
		true_false(b, :) = data(i, :);
		b = b + 1;
	elseif (data.output{i} == "False" & data.predict{i} == "True")
		false_true(c, :) = data(i, :);
		c = c + 1;
	elseif (data.output{i} == "False" & data.predict{i} == "False")
		false_false(d, :) = data(i, :);
		d = d + 1;
	end
end
out = [true_true; true_false; false_true; false_false];
% Changes prefix of the file.
t = char(location);
t(1:strfind(location, "_")) = '';
loc = string(t);
s = "../data/mismatched_" + location;

writetable(out, s, 'Delimiter', ',');

end