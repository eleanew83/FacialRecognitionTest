#!/usr/bin/env python3
"""
Script to extract metadata from Gibraltar Macaque photos and save to Excel.
Extracts information from filename patterns and EXIF data.
"""

import os
import re
import pandas as pd
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
import datetime
from pathlib import Path
import argparse

def parse_filename(filename):
    """
    Parse filename to extract structured information.
    Expected pattern: DDMMYY location sex additional_info observer_name.JPG
    Example: 071224 MH AF Behar INFF Bouny wpt4447 Sylv1.JPG
    """
    # Remove file extension
    name_without_ext = os.path.splitext(filename)[0]
    
    # Split by spaces
    parts = name_without_ext.split()
    
    parsed_info = {
        'filename': filename,
        'date_code': '',
        'location_code': '',
        'sex_code': '',
        'additional_info': '',
        'observer_name': '',
        'parsed_date': '',
        'waypoint_number': ''
    }
    
    if len(parts) >= 3:
        # First part should be date (DDMMYY)
        date_match = re.match(r'(\d{6})', parts[0])
        if date_match:
            parsed_info['date_code'] = parts[0]
            # Try to parse date
            try:
                day = int(parts[0][:2])
                month = int(parts[0][2:4])
                year = int(parts[0][4:6])
                # Assume 21st century for 2-digit years
                if year < 50:
                    year += 2000
                else:
                    year += 1900
                parsed_info['parsed_date'] = f"{year:04d}-{month:02d}-{day:02d}"
            except:
                parsed_info['parsed_date'] = 'Invalid date'
        
        # Second part should be location code
        if len(parts) >= 2:
            parsed_info['location_code'] = parts[1]
        
        # Extract sex/age code - only if there's exactly one valid code
        valid_codes = []
        sex_age_pattern = re.compile(r'^(A?[MF]|S[AM]M?)$')  # Matches AM, AF, SAM, SM, etc.
        
        for part in parts[2:]:
            if sex_age_pattern.match(part):
                valid_codes.append(part)
        
        # Only set sex_code if there's exactly one valid code found
        if len(valid_codes) == 1:
            parsed_info['sex_code'] = valid_codes[0]
        
        # Extract waypoint number from anywhere in the filename
        waypoint_match = re.search(r'wpt(\d+)', name_without_ext)
        if waypoint_match:
            parsed_info['waypoint_number'] = waypoint_match.group(1)
        
        # Extract observer name from the last part (before any numbers/suffixes)
        if len(parts) > 3:
            # Look at the last meaningful part
            last_part = parts[-1]
            # Remove trailing numbers and underscores to get observer name
            observer_match = re.match(r'([A-Za-z]+)', last_part)
            if observer_match:
                parsed_info['observer_name'] = observer_match.group(1)
        
        # Middle parts (after sex code, before observer) are additional info
        if len(parts) > 4:
            # Skip the first 3 parts (date, location, sex) and exclude the last part if it's just the observer
            middle_parts = parts[3:]
            # If we extracted an observer name, we might want to keep the full last part in additional info
            parsed_info['additional_info'] = ' '.join(middle_parts)
        elif len(parts) == 4:
            parsed_info['additional_info'] = parts[3]
    
    return parsed_info

def extract_gps_info(exif_data):
    """
    Extract GPS coordinates from EXIF data and convert to decimal degrees.
    Returns None if no valid GPS coordinates are found.
    """
    gps_info = {
        'latitude': None,
        'longitude': None,
        'altitude': None
    }
    
    # Look for GPS data in the EXIF tags
    gps_data = None
    
    # Check by tag name
    if 'GPSInfo' in exif_data:
        gps_data = exif_data['GPSInfo']
    
    # Check by tag ID
    if not gps_data and 34853 in exif_data:
        gps_data = exif_data[34853]
    
    # No GPS data found
    if not gps_data:
        return gps_info
    
    # Check if we have the required tags for actual coordinates
    # We need tags 1, 2, 3, 4 at minimum (lat ref, lat, long ref, long)
    if not (isinstance(gps_data, dict) and 
            1 in gps_data and 2 in gps_data and 
            3 in gps_data and 4 in gps_data):
        # This is just a placeholder without actual coordinates
        return gps_info
    
    try:
        # Extract latitude
        lat_ref = gps_data[1]
        lat = gps_data[2]
        if lat and lat_ref:
            lat_value = convert_to_decimal_degrees(lat, lat_ref)
            gps_info['latitude'] = lat_value
        
        # Extract longitude
        lon_ref = gps_data[3]
        lon = gps_data[4]
        if lon and lon_ref:
            lon_value = convert_to_decimal_degrees(lon, lon_ref)
            gps_info['longitude'] = lon_value
        
        # Extract altitude
        if 6 in gps_data:
            try:
                altitude = float(gps_data[6])
                # Check if it's below sea level (GPSAltitudeRef = 1)
                if 5 in gps_data and gps_data[5] == 1:
                    altitude = -altitude
                gps_info['altitude'] = round(altitude, 2)
            except (ValueError, TypeError):
                pass
    except Exception:
        # Any error means we couldn't extract valid coordinates
        pass
    
    return gps_info

def convert_to_decimal_degrees(gps_coords, gps_ref):
    """
    Convert GPS coordinates from degrees/minutes/seconds to decimal degrees.
    """
    if not gps_coords or len(gps_coords) != 3:
        return None
    
    try:
        degrees = float(gps_coords[0])
        minutes = float(gps_coords[1])
        seconds = float(gps_coords[2])
        
        decimal_degrees = degrees + (minutes / 60.0) + (seconds / 3600.0)
        
        # Apply direction (South and West are negative)
        if gps_ref in ['S', 'W']:
            decimal_degrees = -decimal_degrees
            
        return round(decimal_degrees, 6)
    except (ValueError, TypeError, ZeroDivisionError):
        return None

def get_exif_data(image_path):
    """
    Extract EXIF data from image.
    """
    exif_data = {}
    
    try:
        with Image.open(image_path) as image:
            exif_dict = image._getexif()
            
            if exif_dict is not None:
                for tag_id, value in exif_dict.items():
                    tag = TAGS.get(tag_id, tag_id)
                    exif_data[tag] = value
    except Exception as e:
        print(f"Error reading EXIF from {image_path}: {e}")
    
    return exif_data

def extract_useful_exif(exif_data):
    """
    Extract the most useful EXIF fields for our purposes.
    """
    useful_fields = {
        'camera_make': exif_data.get('Make', ''),
        'camera_model': exif_data.get('Model', ''),
        'datetime_original': exif_data.get('DateTimeOriginal', ''),
        'datetime_digitized': exif_data.get('DateTimeDigitized', ''),
        'focal_length': exif_data.get('FocalLength', ''),
        'exposure_time': exif_data.get('ExposureTime', ''),
        'f_number': exif_data.get('FNumber', ''),
        'iso': exif_data.get('ISOSpeedRatings', ''),
        'flash': exif_data.get('Flash', ''),
        'orientation': exif_data.get('Orientation', ''),
        'resolution_x': exif_data.get('XResolution', ''),
        'resolution_y': exif_data.get('YResolution', ''),
        'software': exif_data.get('Software', ''),
    }
    
    # Convert datetime strings to proper format if they exist
    for dt_field in ['datetime_original', 'datetime_digitized']:
        if useful_fields[dt_field]:
            try:
                # EXIF datetime format: "YYYY:MM:DD HH:MM:SS"
                dt_obj = datetime.datetime.strptime(useful_fields[dt_field], "%Y:%m:%d %H:%M:%S")
                useful_fields[dt_field] = dt_obj.strftime("%Y-%m-%d %H:%M:%S")
            except:
                pass  # Keep original string if parsing fails
    
    return useful_fields

def get_file_stats(file_path):
    """
    Get file system statistics.
    """
    try:
        stat = os.stat(file_path)
        return {
            'file_size_bytes': stat.st_size,
            'file_size_mb': round(stat.st_size / (1024 * 1024), 2),
            'file_modified': datetime.datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
            'file_created': datetime.datetime.fromtimestamp(stat.st_ctime).strftime("%Y-%m-%d %H:%M:%S")
        }
    except:
        return {
            'file_size_bytes': '',
            'file_size_mb': '',
            'file_modified': '',
            'file_created': ''
        }

def process_directory(base_path, group_name):
    """
    Process all images in the given directory structure.
    """
    all_data = []
    
    # Process both males and females
    for sex_dir in ['females', 'males']:
        sex_path = os.path.join(base_path, sex_dir)
        
        if not os.path.exists(sex_path):
            print(f"Directory not found: {sex_path}")
            continue
            
        print(f"Processing {sex_dir}...")
        
        # Get all individual directories
        individual_dirs = [d for d in os.listdir(sex_path) 
                          if os.path.isdir(os.path.join(sex_path, d)) and not d.startswith('.')]
        
        for individual_dir in individual_dirs:
            individual_path = os.path.join(sex_path, individual_dir)
            print(f"  Processing individual: {individual_dir}")
            
            # Get all image files, excluding files that start with "."
            image_extensions = {'.jpg', '.jpeg', '.png', '.tiff', '.tif'}
            image_files = [f for f in os.listdir(individual_path) 
                          if (os.path.splitext(f.lower())[1] in image_extensions 
                              and not f.startswith('.'))]
            
            for image_file in image_files:
                image_path = os.path.join(individual_path, image_file)
                
                # Parse filename
                filename_info = parse_filename(image_file)
                
                # Get EXIF data
                exif_data = get_exif_data(image_path)
                exif_useful = extract_useful_exif(exif_data)
                
                # Extract GPS information (only actual coordinates, not placeholders)
                gps_info = extract_gps_info(exif_data)
                
                # Get file stats
                file_stats = get_file_stats(image_path)
                
                # Combine all data
                row_data = {
                    'filename': filename_info['filename'],
                    'individual_name': individual_dir,  # Use directory name as the true individual name
                    'group': group_name,
                    'sex': sex_dir.rstrip('s'),  # 'females' -> 'female', 'males' -> 'male'
                    'date_from_filename': filename_info['parsed_date'],
                    'location_code': filename_info['location_code'],
                    'sex_code': filename_info['sex_code'],
                    'waypoint_number': filename_info['waypoint_number'],
                    'observer_name': filename_info['observer_name'],
                    'additional_notes': filename_info['additional_info'],
                    'file_path': image_path,
                    
                    # GPS/Location data (will be None if no actual coordinates)
                    'latitude': gps_info['latitude'],
                    'longitude': gps_info['longitude'],
                    'altitude_meters': gps_info['altitude'],
                    
                    # EXIF data
                    'exif_datetime_original': exif_useful['datetime_original'],
                    'exif_datetime_digitized': exif_useful['datetime_digitized'],
                    'camera_make': exif_useful['camera_make'],
                    'camera_model': exif_useful['camera_model'],
                    'focal_length': exif_useful['focal_length'],
                    'exposure_time': exif_useful['exposure_time'],
                    'f_number': exif_useful['f_number'],
                    'iso': exif_useful['iso'],
                    'flash': exif_useful['flash'],
                    'orientation': exif_useful['orientation'],
                    'resolution_x': exif_useful['resolution_x'],
                    'resolution_y': exif_useful['resolution_y'],
                    'software': exif_useful['software'],
                    
                    # File stats
                    'file_size_mb': file_stats['file_size_mb'],
                    'file_size_bytes': file_stats['file_size_bytes'],
                    'file_modified': file_stats['file_modified'],
                    'file_created': file_stats['file_created'],
                    
                    # Notes column for manual annotations
                    'notes': ''
                }
                
                all_data.append(row_data)
                print(f"    Processed: {image_file}")
    
    return all_data

def main():
    parser = argparse.ArgumentParser(description='Extract metadata from Gibraltar Macaque photos')
    parser.add_argument('--base-path', 
                       default='/home/ylj20/Gibraltar_Macaques_Photos_Cleaned',
                       help='Base path to the directory containing macaque photos')
    parser.add_argument('--group',
                       default='Middle Hill',
                       help='Name of the macaque group being processed')
    parser.add_argument('--output',
                       default=None,
                       help='Output Excel file name (defaults to <group_name>_macaque_metadata.xlsx)')
    
    args = parser.parse_args()
    
    # Construct the full path by joining base_path with group name
    # Keep the original group name with spaces for directory path
    full_path = os.path.join(args.base_path, args.group)
    
    if not os.path.exists(full_path):
        print(f"Error: Path does not exist: {full_path}")
        return
    
    # Set the output filename based on group name if not explicitly provided
    if args.output is None:
        args.output = f"{args.group.replace(' ', '_').lower()}_macaque_metadata.xlsx"
    
    print(f"Starting metadata extraction from: {full_path}")
    print(f"Group name: {args.group}")
    print(f"Output will be saved to: {args.output}")
    
    # Process all images
    all_data = process_directory(full_path, args.group)
    
    if not all_data:
        print("No images found to process!")
        return
    
    # Create DataFrame
    df = pd.DataFrame(all_data)
    
    # Sort by individual name and filename
    df = df.sort_values(['individual_name', 'filename'])
    
    # Function to clean strings for Excel
    def clean_for_excel(value):
        if isinstance(value, str):
            # Replace any characters that Excel might have trouble with
            # This includes control characters, especially \x00-\x1F except for \t, \r, \n
            result = ''.join(ch if ch >= ' ' and ch != '\x7f' else ' ' for ch in value)
            return result
        return value
    
    # Apply cleaning to all string columns in the DataFrame
    for column in df.select_dtypes(include=['object']).columns:
        df[column] = df[column].apply(clean_for_excel)
    
    # Save to Excel
    with pd.ExcelWriter(args.output, engine='openpyxl') as writer:
        # Main sheet with all data
        df.to_excel(writer, sheet_name='All Images', index=False)
        
        # Summary sheet
        summary_data = []
        for individual in df['individual_name'].unique():
            individual_df = df[df['individual_name'] == individual]
            # Count images with GPS coordinates
            gps_count = len(individual_df.dropna(subset=['latitude', 'longitude']))
            # Count images with waypoints
            waypoint_count = len(individual_df[individual_df['waypoint_number'] != ''])
            summary_data.append({
                'Individual': individual,
                'Sex': individual_df['sex'].iloc[0],
                'Group': individual_df['group'].iloc[0],
                'Total Images': len(individual_df),
                'Images with GPS': gps_count,
                'Images with Waypoints': waypoint_count,
                'Date Range': f"{individual_df['date_from_filename'].min()} to {individual_df['date_from_filename'].max()}",
                'Total Size (MB)': round(individual_df['file_size_mb'].sum(), 2)
            })
        
        summary_df = pd.DataFrame(summary_data)
        # Clean summary data strings too
        for column in summary_df.select_dtypes(include=['object']).columns:
            summary_df[column] = summary_df[column].apply(clean_for_excel)
            
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        # Waypoint analysis sheet
        waypoint_data = df[df['waypoint_number'] != ''].copy()
        if len(waypoint_data) > 0:
            waypoint_summary = waypoint_data.groupby('waypoint_number').agg({
                'filename': 'count',
                'individual_name': lambda x: ', '.join(sorted(set(x))),
                'date_from_filename': ['min', 'max']
            }).reset_index()
            waypoint_summary.columns = ['Waypoint', 'Image Count', 'Individuals', 'First Date', 'Last Date']
            
            # Clean waypoint summary strings too
            for column in waypoint_summary.select_dtypes(include=['object']).columns:
                waypoint_summary[column] = waypoint_summary[column].apply(clean_for_excel)
                
            waypoint_summary.to_excel(writer, sheet_name='Waypoint Analysis', index=False)
    
    print(f"\nMetadata extraction complete!")
    print(f"Processed {len(all_data)} images from {len(df['individual_name'].unique())} individuals")
    print(f"Output saved to: {args.output}")
    
    # Print some basic statistics
    print(f"\nBasic Statistics:")
    print(f"Total individuals: {len(df['individual_name'].unique())}")
    print(f"Females: {len(df[df['sex'] == 'female']['individual_name'].unique())}")
    print(f"Males: {len(df[df['sex'] == 'male']['individual_name'].unique())}")
    print(f"Total file size: {df['file_size_mb'].sum():.2f} MB")
    
    # GPS statistics
    gps_images = df.dropna(subset=['latitude', 'longitude'])
    print(f"\nGPS Statistics:")
    print(f"Images with GPS coordinates: {len(gps_images)} ({len(gps_images)/len(df)*100:.1f}%)")
    if len(gps_images) > 0:
        print(f"Latitude range: {gps_images['latitude'].min():.6f} to {gps_images['latitude'].max():.6f}")
        print(f"Longitude range: {gps_images['longitude'].min():.6f} to {gps_images['longitude'].max():.6f}")
    
    # Waypoint statistics
    waypoint_images = df[df['waypoint_number'] != '']
    print(f"\nWaypoint Statistics:")
    print(f"Images with waypoint numbers: {len(waypoint_images)} ({len(waypoint_images)/len(df)*100:.1f}%)")
    if len(waypoint_images) > 0:
        unique_waypoints = waypoint_images['waypoint_number'].nunique()
        print(f"Unique waypoints: {unique_waypoints}")
        print(f"Waypoint range: {waypoint_images['waypoint_number'].min()} to {waypoint_images['waypoint_number'].max()}")

if __name__ == "__main__":
    main() 