def sortBands(bands):
    # Exclude 'last_pos' from sorting by creating a filtered dictionary without this key
    filtered_bands = {k: v for k, v in bands.items() if k != 'last_pos'}
    # Sort the dictionary by keys (x-coordinates), and convert it back to a list of tuples
    sorted_bands = sorted(filtered_bands.items(), key=lambda x: x[0])
    return sorted_bands

# Example bands dictionary
bands = {224: ('BLACK', (224, 99)), 166: ('BLACK', (166, 99)), 109: ('BROWN', (109, 99)), 327: ('ORANGE', (327, 99)), 272: ('ORANGE', (272, 99))}

# Call the function to sort bands
sorted_bands = sortBands(bands)
print("Sorted bands:", sorted_bands)
