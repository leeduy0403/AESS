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

export default function CreateCourses() {
  const [subjects, setSubjects] = useState([]);
  const [selectedSubjects, setSelectedSubjects] = useState([]);
  const [selectedYears, setSelectedYears] = useState([]);
  const [selectedSemesters, setSelectedSemesters] = useState([]);

  const [createError, setCreateError] = useState(null);
  const [createSuccess, setCreateSuccess] = useState(null);
  const [loading, setLoading] = useState(false);

  const academicYears = [
    "2021 - 2022",
    "2022 - 2023",
    "2023 - 2024",
    "2024 - 2025",
    "2025 - 2026",
    "2026 - 2027",
  ];

  const semesters = [1, 2, 3];

  useEffect(() => {
    const fetchSubjects = async () => {
      try {
        const res = await fetch("/api/subject/get");
        const data = await res.json();
        setSubjects(data);
      } catch (error) {
        console.error(error);
      }
    };
    fetchSubjects();
  }, []);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setCreateError(null);
    setCreateSuccess(null);
    const courses = [];
    selectedSubjects?.forEach((subjectId) => {
      selectedYears?.forEach((yearRange) => {
        const [startYear, endYear] = yearRange.split("-");
        selectedSemesters?.forEach((sem) => {
          const shortYear = startYear.slice(2);
          const finalSemester = parseInt(shortYear + sem);
          courses.push({
            subjectId,
            startAcademicYear: parseInt(startYear),
            endAcademicYear: parseInt(endYear),
            semester: finalSemester,
          });
        });
      });
    });
    if (courses?.length === 0) {
      setCreateError(
        "Please select at least one subject, academic year, and semester."
      );
      return;
    }
    try {
      const res = await fetch("/api/course/create-multiple", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ courses }),
      });
      const data = await res.json();
      if (res.ok) {
        setCreateSuccess(`${data?.length} courses created successfully!`);
        setSelectedSubjects([]);
        setSelectedYears([]);
        setSelectedSemesters([]);
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
      <h1 className="my-8 text-center font-bold text-4xl">Create Courses</h1>
      <form className="flex flex-col gap-6" onSubmit={handleSubmit}>
        <div className="flex flex-col gap-2">
          <Label value="Subjects" className="text-lg" />
          <FormControl fullWidth>
            <InputLabel id="label">Subjects</InputLabel>
            <Select
              labelId="label"
              label="Subjects"
              multiple
              value={selectedSubjects}
              onChange={(e) => setSelectedSubjects(e.target.value)}
              renderValue={(selected) => (
                <div className="flex flex-wrap gap-1">
                  {selected?.length > 0 &&
                    selected.map((id) => {
                      const subject = subjects.find((s) => s?._id === id);
                      return <Chip key={id} label={subject?.name || id} />;
                    })}
                </div>
              )}
            >
              {subjects?.length > 0 &&
                subjects.map((subject) => (
                  <MenuItem key={subject?._id} value={subject?._id}>
                    {subject?.code}_{subject?.name}
                  </MenuItem>
                ))}
            </Select>
          </FormControl>
        </div>
        <div className="flex flex-col gap-2">
          <Label value="Academic Years" className="text-lg" />
          <FormControl fullWidth>
            <InputLabel id="label">Academic Years</InputLabel>
            <Select
              labelId="label"
              label="Academic Years"
              multiple
              value={selectedYears}
              onChange={(e) => setSelectedYears(e.target.value)}
              renderValue={(selected) => (
                <div className="flex flex-wrap gap-1">
                  {selected?.length > 0 &&
                    selected.map((year) => <Chip key={year} label={year} />)}
                </div>
              )}
            >
              {academicYears?.length > 0 &&
                academicYears.map((year) => (
                  <MenuItem key={year} value={year}>
                    {year}
                  </MenuItem>
                ))}
            </Select>
          </FormControl>
        </div>
        <div className="flex flex-col gap-2">
          <Label value="Semesters" className="text-lg" />
          <FormControl fullWidth>
            <InputLabel id="label">Semesters</InputLabel>
            <Select
              labelId="label"
              label="Semesters"
              multiple
              value={selectedSemesters}
              onChange={(e) => setSelectedSemesters(e.target.value)}
              renderValue={(selected) => (
                <div className="flex flex-wrap gap-1">
                  {selected?.length > 0 &&
                    selected.map((sem) => (
                      <Chip key={sem} label={`Semester ${sem}`} />
                    ))}
                </div>
              )}
            >
              {semesters?.length > 0 &&
                semesters.map((sem) => (
                  <MenuItem key={sem} value={sem}>
                    Semester {sem}
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
          Create Courses
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
