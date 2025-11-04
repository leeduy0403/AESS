import {
  Alert,
  Button,
  FormControl,
  InputLabel,
  MenuItem,
  Select,
} from "@mui/material";
import { Label, TextInput } from "flowbite-react";
import { useEffect, useState } from "react";

export default function DashCreateSubject() {
  const [faculties, setFaculties] = useState([]);
  const [formData, setFormData] = useState({
    facultyId: "",
    name: "",
    code: "",
  });
  const [createError, setCreateError] = useState(null);
  const [createSuccess, setCreateSuccess] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    const fetchFaculties = async () => {
      try {
        const res = await fetch(`/api/faculty/get`);
        const data = await res.json();
        if (res.ok) {
          setFaculties(data);
        }
      } catch (error) {
        console.log(error.message);
      }
    };
    fetchFaculties();
  }, []);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setCreateError(null);
    setCreateSuccess(null);
    try {
      const res = await fetch("/api/subject/create", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(formData),
      });
      const data = await res.json();
      if (res.ok) {
        setCreateSuccess("Create subject successfully!");
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
      <h1 className="my-8 text-center font-bold text-4xl">Create Subject</h1>
      <form className="flex flex-col gap-4" onSubmit={handleSubmit}>
        <div className="flex flex-col gap-2">
          <Label value="Faculty" className="text-lg" />
          <FormControl fullWidth>
            <InputLabel id="label">Choose a faculty</InputLabel>
            <Select
              labelId="label"
              label="Choose a faculty"
              value={formData?.facultyId}
              onChange={(e) =>
                setFormData({ ...formData, facultyId: e.target.value })
              }
            >
              {faculties?.length > 0 &&
                faculties.map((faculty) => (
                  <MenuItem key={faculty?._id} value={faculty?._id}>
                    {faculty?.name}
                  </MenuItem>
                ))}
            </Select>
          </FormControl>
        </div>
        <div className="flex flex-col gap-2">
          <Label value="Name" className="text-lg" />
          <TextInput
            placeholder="Enter subject name"
            required
            id="name"
            sizing="lg"
            onChange={(e) => setFormData({ ...formData, name: e.target.value })}
          />
        </div>
        <div className="flex flex-col gap-2">
          <Label value="Code" className="text-lg" />
          <TextInput
            placeholder="Enter course code"
            required
            id="code"
            sizing="lg"
            onChange={(e) => setFormData({ ...formData, code: e.target.value })}
          />
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
          Create Subject
        </Button>
        {createError && (
          <Alert severity="error" className=" border border-red-600">
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
