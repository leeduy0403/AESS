import { useState, useEffect } from "react";
import {
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Chip,
  Button,
  Alert,
} from "@mui/material";
import { Label } from "flowbite-react";

export default function CreateClasses() {
  const [courses, setCourses] = useState([]);
  const [selectedCourses, setSelectedCourses] = useState([]);
  const [selectedClassNames, setSelectedClassNames] = useState([]);
  const [createError, setCreateError] = useState(null);
  const [createSuccess, setCreateSuccess] = useState(null);
  const [loading, setLoading] = useState(false);
  const classNames = [
    "CC01",
    "CC02",
    "CC03",
    "CC04",
    "CC05",
    "CC06",
    "CC07",
    "CC08",
    "CC09",
    "CC10",
  ];

  useEffect(() => {
    const fetchCourses = async () => {
      try {
        const res = await fetch("/api/course/get");
        const data = await res.json();
        setCourses(data);
      } catch (error) {
        console.error(error);
      }
    };
    fetchCourses();
  }, []);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setCreateError(null);
    setCreateSuccess(null);
    const classes = [];
    selectedCourses?.forEach((courseId) => {
      selectedClassNames?.forEach((className) => {
        classes.push({
          courseId,
          name: className,
        });
      });
    });
    if (classes?.length === 0) {
      setCreateError("Please select at least one course and class name.");
      return;
    }
    try {
      const res = await fetch("/api/class/create-multiple", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ classes }),
      });
      const data = await res.json();
      if (res.ok) {
        setCreateSuccess(`${data?.length} classes created successfully!`);
        setSelectedCourses([]);
        setSelectedClassNames([]);
      } else {
        setCreateError(data.message);
      }
      setLoading(false);
    } catch (error) {
      setCreateError(error.message);
      setLoading(false);
    }
  };

  return (
    <div className="max-w-4xl mx-auto p-4 w-full">
      <h1 className="my-8 text-center font-bold text-4xl">Create Classes</h1>
      <form className="flex flex-col gap-6" onSubmit={handleSubmit}>
        <div className="flex flex-col gap-2">
          <Label value="Courses" className="text-lg" />
          <FormControl fullWidth>
            <InputLabel id="label">Courses</InputLabel>
            <Select
              labelId="label"
              label="Courses"
              multiple
              value={selectedCourses}
              onChange={(e) => setSelectedCourses(e.target.value)}
              renderValue={(selected) => (
                <div className="flex flex-wrap gap-1">
                  {selected?.length > 0 &&
                    selected.map((id) => {
                      const course = courses?.find((c) => c?._id === id);
                      return (
                        <Chip
                          key={id}
                          label={course ? course?.subjectId?.name || id : id}
                        />
                      );
                    })}
                </div>
              )}
            >
              {courses?.length > 0 &&
                courses.map((course) => (
                  <MenuItem key={course?._id} value={course?._id}>
                    {course?.subjectId?.code}_{course?.subjectId?.name}
                  </MenuItem>
                ))}
            </Select>
          </FormControl>
        </div>
        <div className="flex flex-col gap-2">
          <Label value="Class Names" className="text-lg" />
          <FormControl fullWidth>
            <InputLabel id="label">Class Names</InputLabel>
            <Select
              labelId="label"
              label=">Class Names"
              multiple
              value={selectedClassNames}
              onChange={(e) => setSelectedClassNames(e.target.value)}
              renderValue={(selected) => (
                <div className="flex flex-wrap gap-1">
                  {selected?.length > 0 &&
                    selected.map((name) => <Chip key={name} label={name} />)}
                </div>
              )}
            >
              {classNames?.length > 0 &&
                classNames.map((name) => (
                  <MenuItem key={name} value={name}>
                    {name}
                  </MenuItem>
                ))}
            </Select>
          </FormControl>
        </div>
        <Button
          variant="contained"
          style={{
            backgroundColor: "#26597C",
            color: "#FFFFFF",
            textTransform: "none",
          }}
          type="submit"
          size="large"
          disabled={loading}
        >
          Create Classes
        </Button>
        {createError && (
          <Alert severity="error" className="border border-red-600">
            {createError}
          </Alert>
        )}
        {createSuccess && (
          <Alert severity="success" className="border border-green-600">
            {createSuccess}
          </Alert>
        )}
      </form>
    </div>
  );
}
