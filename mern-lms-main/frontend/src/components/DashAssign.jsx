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

export default function DashAssign() {
  const [classes, setClasses] = useState([]);
  const [educators, setEducators] = useState([]);
  const [students, setStudents] = useState([]);
  const [selectedClassIds, setSelectedClassIds] = useState([]);
  const [selectedEducatorIds, setSelectedEducatorIds] = useState([]);
  const [selectedStudentIds, setSelectedStudentIds] = useState([]);
  const [assignError, setAssignError] = useState(null);
  const [assignSuccess, setAssignSuccess] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const [classesRes, educatorsRes, studentsRes] = await Promise.all([
          fetch("/api/class/get"),
          fetch("/api/user/get-educators"),
          fetch("/api/user/get-students"),
        ]);
        const [classesData, educatorsData, studentsData] = await Promise.all([
          classesRes.json(),
          educatorsRes.json(),
          studentsRes.json(),
        ]);
        setClasses(classesData);
        setEducators(educatorsData);
        setStudents(studentsData);
      } catch (error) {
        console.error(error);
      }
    };
    fetchData();
  }, []);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setAssignError(null);
    setAssignSuccess(null);
    try {
      const res = await fetch(`/api/class/assign`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          classIds: selectedClassIds,
          educatorIds: selectedEducatorIds,
          studentIds: selectedStudentIds,
        }),
      });
      const data = await res.json();
      if (res.ok) {
        setAssignSuccess("Assignment successful!");
        setSelectedClassIds([]);
        setSelectedEducatorIds([]);
        setSelectedStudentIds([]);
      } else {
        setAssignError(data.message);
      }
      setLoading(false);
    } catch (error) {
      setAssignError(error.message);
      setLoading(false);
    }
  };

  return (
    <div className="max-w-5xl mx-auto p-4 w-full">
      <h1 className="my-8 text-center font-bold text-4xl">Assign Classes</h1>
      <form className="flex flex-col gap-6" onSubmit={handleSubmit}>
        <div className="flex flex-col gap-2">
          <Label value="Class Names" className="text-lg" />
          <FormControl fullWidth>
            <InputLabel id="label">Classes</InputLabel>
            <Select
              labelId="label"
              label="Classes"
              multiple
              value={selectedClassIds}
              onChange={(e) => setSelectedClassIds(e.target.value)}
              renderValue={(selected) => (
                <div className="flex flex-wrap gap-1">
                  {selected?.length > 0 &&
                    selected.map((id) => {
                      const classItem = classes.find((c) => c?._id === id);
                      return <Chip key={id} label={classItem?.name || id} />;
                    })}
                </div>
              )}
            >
              {classes?.length > 0 &&
                classes.map((cls) => (
                  <MenuItem key={cls?._id} value={cls?._id}>
                    {cls?.courseId?.subjectId?.code}_Academic Year{" "}
                    {cls?.courseId?.startAcademicYear}-
                    {cls?.courseId?.endAcademicYear}_Semester{" "}
                    {cls?.semester % 10}_{cls?.name}_
                    {cls?.courseId?.subjectId?.name}
                  </MenuItem>
                ))}
            </Select>
          </FormControl>
        </div>
        <div className="flex flex-col gap-2">
          <Label value="Educator Names" className="text-lg" />
          <FormControl fullWidth>
            <InputLabel id="label">Educators</InputLabel>
            <Select
              labelId="label"
              label="Educators"
              multiple
              value={selectedEducatorIds}
              onChange={(e) => setSelectedEducatorIds(e.target.value)}
              renderValue={(selected) => (
                <div className="flex flex-wrap gap-1">
                  {selected?.length > 0 &&
                    selected.map((id) => {
                      const educator = educators?.find((e) => e?._id === id);
                      return <Chip key={id} label={educator?.name || id} />;
                    })}
                </div>
              )}
            >
              {educators?.length > 0 &&
                educators.map((edu) => (
                  <MenuItem key={edu?._id} value={edu?._id}>
                    {edu?.educatorId}_{edu?.name}
                  </MenuItem>
                ))}
            </Select>
          </FormControl>
        </div>
        <div className="flex flex-col gap-2">
          <Label value="Student Names" className="text-lg" />
          <FormControl fullWidth>
            <InputLabel id="label">Students</InputLabel>
            <Select
              labelId="label"
              label="Students"
              multiple
              value={selectedStudentIds}
              onChange={(e) => setSelectedStudentIds(e.target.value)}
              renderValue={(selected) => (
                <div className="flex flex-wrap gap-1">
                  {selected?.length > 0 &&
                    selected.map((id) => {
                      const student = students.find((s) => s?._id === id);
                      return <Chip key={id} label={student?.name || id} />;
                    })}
                </div>
              )}
            >
              {students?.length > 0 &&
                students.map((stu) => (
                  <MenuItem key={stu?._id} value={stu?._id}>
                    {stu?.studentId}_{stu?.name}
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
          Save Assign Course
        </Button>
        {assignError && (
          <Alert severity="error" className="border border-red-600">
            {assignError}
          </Alert>
        )}
        {assignSuccess && (
          <Alert severity="success" className="border border-green-600">
            {assignSuccess}
          </Alert>
        )}
      </form>
    </div>
  );
}
