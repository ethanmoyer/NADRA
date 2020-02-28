% This function accepts the location of a table object (.csv) and creates 
% two arrays of equal size: one contains only true outputs and the other 
% contains only false outputs. The result is a new table object (.csv) with
% the two concatenated together with the prefix 'split_' saved under the
% ../data/ directory.

function currateData(loc)
data = readtable(loc);
data = rmmissing(data);
size(data, 1);
T = [];
F = [];
L = [1:size(data, 1)];
for i = L
	if data.output{i} == "TRUE"
		T(i, 1) = true;
		F(i, 1) = false;
	else
		T(i, 1) = false;
		F(i, 1) = true;
	end
end

T = logical(T);
F = logical(F);
true_output = data(L(T), :);
false_output = data(L(F), :);

size_true = size(true_output, 1);
size_false = size(false_output, 1);

if size_true < size_false
	equal_false = false_output(randperm(size_false, size_true), :);
	table_out = [true_output; equal_false];
else
	equal_true = true_output(randperm(size_true, size_false), :);
	table_out = [false_output; equal_true];
end

% Changes prefix of the file.
t = char(loc);
t(1:strfind(loc, "_")) = '';
loc = string(t);
s = "../data/split_" + loc;

writetable(table_out, s, 'Delimiter', ',')

end