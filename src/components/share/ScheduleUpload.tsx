import React, { useState } from 'react';
import { Calendar } from 'lucide-react';
import DatePicker from 'react-datepicker';
import "react-datepicker/dist/react-datepicker.css";

interface ScheduleUploadProps {
  onSchedule: (date: Date | null) => void;
  scheduledDate: Date | null;
}

export const ScheduleUpload: React.FC<ScheduleUploadProps> = ({ onSchedule, scheduledDate }) => {
  const [showPicker, setShowPicker] = useState(false);

  return (
    <div className="relative">
      <button
        onClick={() => setShowPicker(!showPicker)}
        className="inline-flex items-center px-3 py-2 border border-gray-300 shadow-sm text-sm leading-4 font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
      >
        <Calendar className="h-4 w-4 mr-2" />
        {scheduledDate ? new Date(scheduledDate).toLocaleString() : 'Schedule Post'}
      </button>

      {showPicker && (
        <div className="absolute z-10 mt-1">
          <DatePicker
            selected={scheduledDate}
            onChange={(date) => {
              onSchedule(date);
              setShowPicker(false);
            }}
            showTimeSelect
            dateFormat="MMMM d, yyyy h:mm aa"
            minDate={new Date()}
            inline
          />
        </div>
      )}
    </div>
  );
}; 